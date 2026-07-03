import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QStringListModel, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPalette, QPen, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QCompleter,
    QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMainWindow, QPushButton, QSpinBox, QSplitter,
    QTextEdit, QToolBar, QToolTip, QVBoxLayout, QWidget,
)

from document_pipeline import make_document_chain
from file_picker import pick_file
from file_stats import analyze_file, analyze_text
from rag import PROVIDERS, get_models, rag_answer
from session_logger import SessionLogger, load_all_questions
from vector_store import (
    get_document_text, list_indexed_documents, remove_document, store_chain,
)


# ── Spinner widget ────────────────────────────────────────────────────────────

class SpinnerWidget(QWidget):
    def __init__(self, size: int = 32, color: str = "#4a7fc1", parent=None):
        super().__init__(parent)
        self._angle = 0
        self._color = QColor(color)
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def _tick(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(self._color, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        r = self.rect().adjusted(4, 4, -4, -4)
        p.drawArc(r, (-self._angle) * 16, 270 * 16)

    def setVisible(self, visible: bool):
        if visible:
            self._timer.start(16)
        else:
            self._timer.stop()
        super().setVisible(visible)


# ── Background workers ────────────────────────────────────────────────────────

class StatsWorker(QThread):
    done = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.done.emit(analyze_file(self.path))
        except Exception as e:
            self.error.emit(str(e))


class IndexedStatsWorker(QThread):
    done = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, source_name: str):
        super().__init__()
        self.source_name = source_name

    def run(self):
        try:
            text = get_document_text(self.source_name)
            self.done.emit(analyze_text(text))
        except Exception as e:
            self.error.emit(str(e))


class IndexWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, file_path: Path, chunk_size: int, chunk_overlap: int):
        super().__init__()
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def run(self):
        try:
            self.log.emit(f"Indexing {self.file_path.name} …")
            chain = make_document_chain(self.chunk_size, self.chunk_overlap) | store_chain
            n = chain.invoke(self.file_path)
            self.log.emit(f"Stored {n} chunks.")
            self.done.emit(n)
        except Exception as e:
            self.error.emit(str(e))


class AskWorker(QThread):
    log = pyqtSignal(str)
    token = pyqtSignal(str)
    done = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(
        self, query: str, n_results: int, history: list[dict],
        use_web: bool, provider: str, model: str,
    ):
        super().__init__()
        self.query = query
        self.n_results = n_results
        self.history = history
        self.use_web = use_web
        self.provider = provider
        self.model = model

    def run(self):
        try:
            self.log.emit(f"Searching: {self.query}")
            answer, sources = rag_answer(
                self.query,
                n_results=self.n_results,
                on_token=self.token.emit,
                on_retry=self.log.emit,
                history=self.history,
                provider=self.provider,
                model=self.model,
                use_web_fallback=self.use_web,
            )
            self.done.emit(answer, sources)
        except Exception as e:
            self.error.emit(str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _help_btn(tip: str) -> QPushButton:
    btn = QPushButton("?")
    btn.setFixedSize(24, 24)
    btn.setFlat(True)
    btn.setStyleSheet("font-weight: bold; color: #555;")
    btn.setCursor(Qt.CursorShape.WhatsThisCursor)
    btn.clicked.connect(
        lambda: QToolTip.showText(btn.mapToGlobal(btn.rect().bottomLeft()), tip, btn)
    )
    return btn


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document Q&A")
        self.setMinimumSize(1000, 720)
        self._file_path: Path | None = None
        self._history: list[dict] = []
        self._session: SessionLogger | None = None
        self._dark = False
        self._default_palette = QApplication.instance().palette()
        self._stats_worker: StatsWorker | None = None
        self._indexed_stats_worker: IndexedStatsWorker | None = None
        self._index_worker: IndexWorker | None = None
        self._ask_worker: AskWorker | None = None
        self._setup_toolbar()
        self._setup_ui()
        self._refresh_doc_list()

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _setup_toolbar(self):
        tb = QToolBar("Controls")
        tb.setMovable(False)
        tb.setFloatable(False)
        self.addToolBar(tb)

        self.btn_dark = QPushButton("Dark Mode")
        self.btn_dark.setCheckable(True)
        self.btn_dark.toggled.connect(self._on_toggle_dark)
        tb.addWidget(self.btn_dark)

        tb.addSeparator()

        self.btn_clear_hist = QPushButton("Clear History")
        self.btn_clear_hist.setEnabled(False)
        self.btn_clear_hist.clicked.connect(self._on_clear_history)
        tb.addWidget(self.btn_clear_hist)

        tb.addSeparator()

        tb.addWidget(QLabel("  Provider:"))
        self.cb_provider = QComboBox()
        self.cb_provider.addItems(PROVIDERS.keys())
        self.cb_provider.currentTextChanged.connect(self._on_provider_changed)
        tb.addWidget(self.cb_provider)

        tb.addWidget(QLabel("  Model:"))
        self.cb_model = QComboBox()
        self.cb_model.setMinimumWidth(220)
        self.cb_model.addItems(get_models(self.cb_provider.currentText()))
        tb.addWidget(self.cb_model)

    # ── Central layout ────────────────────────────────────────────────────────

    def _setup_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        self.setCentralWidget(splitter)

        splitter.addWidget(self._build_doc_panel())
        splitter.addWidget(self._build_qa_panel())
        splitter.setSizes([200, 800])

    # ── Left panel: document manager ──────────────────────────────────────────

    def _build_doc_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        layout.addWidget(QLabel("Indexed Documents"))

        self.doc_list = QListWidget()
        self.doc_list.currentItemChanged.connect(self._on_doc_selection_changed)
        self.doc_list.itemDoubleClicked.connect(self._on_doc_double_clicked)
        layout.addWidget(self.doc_list, stretch=1)

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._on_remove_doc)
        layout.addWidget(self.btn_remove)

        return panel

    # ── Right panel: Q&A ──────────────────────────────────────────────────────

    def _build_qa_panel(self) -> QWidget:
        panel = QWidget()
        root = QVBoxLayout(panel)
        root.setSpacing(8)
        root.setContentsMargins(14, 14, 14, 14)

        # File picker
        file_row = QHBoxLayout()
        self.btn_pick = QPushButton("Select File")
        self.btn_pick.setFixedWidth(130)
        self.btn_pick.clicked.connect(self._on_pick_file)
        file_info = QVBoxLayout()
        file_info.setSpacing(2)
        self.lbl_file = QLabel("No file selected")
        self.lbl_file.setStyleSheet("color: #888;")
        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color: #666; font-size: 11px;")
        file_info.addWidget(self.lbl_file)
        file_info.addWidget(self.lbl_stats)
        file_row.addWidget(self.btn_pick)
        file_row.addLayout(file_info, stretch=1)
        root.addLayout(file_row)

        # Settings row
        settings = QHBoxLayout()
        settings.setSpacing(6)

        settings.addWidget(QLabel("Chunk:"))
        self.spin_chunk = QSpinBox()
        self.spin_chunk.setRange(100, 8000)
        self.spin_chunk.setValue(1000)
        self.spin_chunk.setSingleStep(50)
        self.spin_chunk.setFixedWidth(80)
        settings.addWidget(self.spin_chunk)
        settings.addWidget(_help_btn(
            "Characters per chunk.\n"
            "Larger = more context per chunk but fewer search hits.\n"
            "Auto-filled based on document size."
        ))

        settings.addSpacing(10)
        settings.addWidget(QLabel("Overlap:"))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 2000)
        self.spin_overlap.setValue(150)
        self.spin_overlap.setSingleStep(50)
        self.spin_overlap.setFixedWidth(80)
        settings.addWidget(self.spin_overlap)
        settings.addWidget(_help_btn(
            "Characters shared between adjacent chunks.\n"
            "Prevents context from being cut at boundaries.\n"
            "Recommended: 10–20% of chunk size."
        ))

        settings.addSpacing(10)
        settings.addWidget(QLabel("Results:"))
        self.spin_results = QSpinBox()
        self.spin_results.setRange(1, 20)
        self.spin_results.setValue(4)
        self.spin_results.setFixedWidth(60)
        settings.addWidget(self.spin_results)
        settings.addWidget(_help_btn(
            "Number of chunks retrieved from the\n"
            "vector store per question.\n"
            "More = broader context; fewer = more focused."
        ))

        settings.addSpacing(10)
        self.chk_web = QCheckBox("Web fallback")
        settings.addWidget(self.chk_web)
        settings.addWidget(_help_btn(
            "If no relevant chunks are found in the document,\n"
            "search DuckDuckGo and answer from web results instead."
        ))

        settings.addStretch()
        root.addLayout(settings)

        # Prompt row
        prompt_row = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Ask a question about the document …")
        self.prompt_input.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.prompt_input.returnPressed.connect(self._on_ask)

        self._history_model = QStringListModel(load_all_questions())
        completer = QCompleter(self._history_model, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.prompt_input.setCompleter(completer)

        self.btn_ask = QPushButton("Ask")
        self.btn_ask.setFixedWidth(70)
        self.btn_ask.setEnabled(False)
        self.btn_ask.clicked.connect(self._on_ask)

        self.spinner = SpinnerWidget(size=32)

        prompt_row.addWidget(self.prompt_input, stretch=1)
        prompt_row.addWidget(self.btn_ask)
        prompt_row.addWidget(self.spinner)
        root.addLayout(prompt_row)

        # Answer
        root.addWidget(QLabel("Answer:"))
        self.answer_box = QTextEdit()
        self.answer_box.setReadOnly(True)
        self.answer_box.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        root.addWidget(self.answer_box, stretch=3)

        # Logs
        root.addWidget(QLabel("Logs:"))
        self.logs_box = QTextEdit()
        self.logs_box.setReadOnly(True)
        self.logs_box.setFixedHeight(110)
        self.logs_box.setStyleSheet("font-family: monospace; font-size: 11px;")
        root.addWidget(self.logs_box)

        return panel

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.logs_box.append(msg)
        self.logs_box.verticalScrollBar().setValue(
            self.logs_box.verticalScrollBar().maximum()
        )

    def _set_busy(self, busy: bool):
        self.btn_pick.setEnabled(not busy)
        self.btn_ask.setEnabled(not busy and self._file_path is not None)
        self.spin_chunk.setEnabled(not busy)
        self.spin_overlap.setEnabled(not busy)
        self.spin_results.setEnabled(not busy)
        self.cb_provider.setEnabled(not busy)
        self.cb_model.setEnabled(not busy)
        self.btn_clear_hist.setEnabled(not busy and bool(self._history))
        self.spinner.setVisible(busy)

    def _refresh_doc_list(self):
        self.doc_list.clear()
        for name in list_indexed_documents():
            self.doc_list.addItem(name)

    def _current_provider(self) -> str:
        return self.cb_provider.currentText()

    def _current_model(self) -> str:
        return self.cb_model.currentText()

    def _append_to_answer(self, token: str):
        if self.spinner.isVisible():
            self.spinner.hide()
        cursor = self.answer_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.answer_box.setTextCursor(cursor)
        self.answer_box.insertPlainText(token)
        self.answer_box.verticalScrollBar().setValue(
            self.answer_box.verticalScrollBar().maximum()
        )

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_pick_file(self):
        self._log("Opening file picker …")
        file_path = pick_file(title="Select a PDF, Word, or Text file")
        if file_path is None:
            self._log("No file selected.")
            return

        self._file_path = file_path
        self.lbl_file.setText(file_path.name)
        self.lbl_file.setStyleSheet("")
        self.lbl_stats.setText("Analysing …")
        self.answer_box.clear()
        self._set_busy(True)

        self._stats_worker = StatsWorker(file_path)
        self._stats_worker.done.connect(self._on_stats_done)
        self._stats_worker.error.connect(self._on_stats_error)
        self._stats_worker.start()

    def _on_stats_done(self, stats: dict):
        chars = f"{stats['char_count']:,}"
        words = f"{stats['word_count']:,}"
        self.lbl_stats.setText(
            f"{stats['size']}  ·  {chars} chars  ·  {words} words  ·  Language: {stats['language']}"
        )
        rec_chunk = stats["recommended_chunk_size"]
        rec_overlap = stats["recommended_overlap"]
        self.spin_chunk.setValue(rec_chunk)
        self.spin_overlap.setValue(rec_overlap)
        self._log(
            f"Stats: {stats['size']}, {chars} chars, {words} words, {stats['language']}"
        )
        self._log(f"Recommended → chunk: {rec_chunk}, overlap: {rec_overlap}")

        self._session = SessionLogger(self._file_path.name)
        self._log(f"Session: {self._session.file_path.name}")

        self._index_worker = IndexWorker(
            self._file_path, chunk_size=rec_chunk, chunk_overlap=rec_overlap,
        )
        self._index_worker.log.connect(self._log)
        self._index_worker.done.connect(self._on_index_done)
        self._index_worker.error.connect(self._on_error)
        self._index_worker.start()

    def _on_stats_error(self, msg: str):
        self.lbl_stats.setText("Could not analyse file.")
        self._log(f"Stats error: {msg}")
        self._set_busy(False)

    def _on_index_done(self, n: int):
        self._log(f"Indexed {n} chunks.")
        self._refresh_doc_list()
        self._set_busy(False)

    def _on_ask(self):
        query = self.prompt_input.text().strip()
        if not query:
            return
        self.answer_box.clear()
        self._log(f"Q: {query}")
        self._set_busy(True)

        self._ask_worker = AskWorker(
            query,
            n_results=self.spin_results.value(),
            history=list(self._history),
            use_web=self.chk_web.isChecked(),
            provider=self._current_provider(),
            model=self._current_model(),
        )
        self._ask_worker.log.connect(self._log)
        self._ask_worker.token.connect(self._append_to_answer)
        self._ask_worker.done.connect(self._on_answer_done)
        self._ask_worker.error.connect(self._on_error)
        self._ask_worker.start()

    def _on_answer_done(self, answer: str, sources: list):
        if sources:
            self._log("Sources: " + " | ".join(s["label"] for s in sources))

        # Conversation history
        query = self._ask_worker.query if self._ask_worker else ""
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": answer})

        # Session log
        if self._session:
            self._session.log(query, answer, sources)
            self._log(f"Saved → {self._session.file_path.name}")

        # Question history dropdown
        current = self._history_model.stringList()
        if query and query not in current:
            self._history_model.setStringList([query] + current)

        self.btn_clear_hist.setEnabled(True)
        self._set_busy(False)

    def _on_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        self._set_busy(False)

    def _on_doc_selection_changed(self, current, _previous):
        self.btn_remove.setEnabled(current is not None)

    def _on_remove_doc(self):
        item = self.doc_list.currentItem()
        if not item:
            return
        name = item.text()
        n = remove_document(name)
        self._log(f"Removed '{name}' ({n} chunks deleted).")
        self._refresh_doc_list()

    def _on_doc_double_clicked(self, item):
        name = item.text()
        self._file_path = Path(name)
        self.lbl_file.setText(name)
        self.lbl_file.setStyleSheet("")
        self.lbl_stats.setText("Analysing …")
        self.answer_box.clear()
        self._session = SessionLogger(name)
        self.btn_ask.setEnabled(False)
        self._log(f"Selected '{name}' from index.")
        self._set_busy(True)

        self._indexed_stats_worker = IndexedStatsWorker(name)
        self._indexed_stats_worker.done.connect(self._on_indexed_stats_done)
        self._indexed_stats_worker.error.connect(self._on_stats_error)
        self._indexed_stats_worker.start()

    def _on_indexed_stats_done(self, stats: dict):
        chars = f"{stats['char_count']:,}"
        words = f"{stats['word_count']:,}"
        self.lbl_stats.setText(
            f"{stats['size']}  ·  {chars} chars  ·  {words} words  ·  Language: {stats['language']}"
        )
        self.spin_chunk.setValue(stats["recommended_chunk_size"])
        self.spin_overlap.setValue(stats["recommended_overlap"])
        self._set_busy(False)
        self.btn_ask.setEnabled(True)

    def _on_provider_changed(self, provider: str):
        self.cb_model.clear()
        self.cb_model.addItems(get_models(provider))

    def _on_clear_history(self):
        self._history.clear()
        self.btn_clear_hist.setEnabled(False)
        self._log("Conversation history cleared.")

    def _on_toggle_dark(self, checked: bool):
        app = QApplication.instance()
        if checked:
            p = QPalette()
            p.setColor(QPalette.ColorRole.Window,          QColor(30,  30,  35))
            p.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
            p.setColor(QPalette.ColorRole.Base,            QColor(40,  40,  45))
            p.setColor(QPalette.ColorRole.AlternateBase,   QColor(50,  50,  55))
            p.setColor(QPalette.ColorRole.ToolTipBase,     QColor(50,  50,  55))
            p.setColor(QPalette.ColorRole.ToolTipText,     QColor(220, 220, 220))
            p.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
            p.setColor(QPalette.ColorRole.Button,          QColor(50,  50,  55))
            p.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
            p.setColor(QPalette.ColorRole.BrightText,      QColor(255, 100, 100))
            p.setColor(QPalette.ColorRole.Link,            QColor(100, 160, 220))
            p.setColor(QPalette.ColorRole.Highlight,       QColor(74,  127, 193))
            p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
            app.setPalette(p)
        else:
            app.setPalette(self._default_palette)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_ui():
    app = QApplication(sys.argv)
    font = QFont("Noto Naskh Arabic", 13)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_ui()
