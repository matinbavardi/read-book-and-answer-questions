"""Run once to produce Architecture.png."""
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

BG = "#f5f6fa"
W, H = 16, 11

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

C_USER  = "#5b8dd9"
C_PROC  = "#48a99a"
C_STORE = "#8e6bbf"
C_LLM   = "#d4834a"
C_LOG   = "#5aaa72"
C_WEB   = "#d45a7a"

BW, BH = 2.3, 0.68


def box(x, y, w, h, title, sub="", fill="#4a7fc1", tc="white"):
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.12", facecolor=fill,
        edgecolor="white", linewidth=2, zorder=3,
    ))
    if sub:
        ax.text(x, y + 0.13, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=tc, zorder=4)
        ax.text(x, y - 0.17, sub, ha="center", va="center",
                fontsize=7, color=tc, alpha=0.88, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color=tc, zorder=4)


def arrow(x1, y1, x2, y2, label="", color="#888"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=2)
    if label:
        mx, my = (x1+x2)/2 + 0.12, (y1+y2)/2
        ax.text(mx, my, label, fontsize=7, color="#666",
                style="italic", va="center", zorder=5)


def curved(x1, y1, x2, y2, rad=0.35, label="", color="#888"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=1.5,
                    connectionstyle=f"arc3,rad={rad}",
                ), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2 + 0.25
        ax.text(mx, my, label, fontsize=7, color="#666",
                style="italic", ha="center", zorder=5)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(W/2, 10.6, "Document Q&A — Architecture",
        ha="center", fontsize=15, fontweight="bold", color="#222")

# ── Phase labels ──────────────────────────────────────────────────────────────
ax.text(3.5, 10.1, "Phase 1 · Indexing", ha="center", fontsize=10,
        color="#444", style="italic")
ax.text(10.5, 10.1, "Phase 2 · Querying", ha="center", fontsize=10,
        color="#444", style="italic")
ax.plot([7, 7], [0.8, 10.4], color="#ddd", lw=1.5, ls="--", zorder=1)


# ── Indexing pipeline ─────────────────────────────────────────────────────────
IX = 3.5
idx = [
    (IX, 9.4,  "File Picker",      "file_picker.py",         C_USER),
    (IX, 8.3,  "Stats Analysis",   "file_stats.py",          C_PROC),
    (IX, 7.2,  "Document Loader",  "document_pipeline.py",   C_PROC),
    (IX, 6.1,  "Text Splitter",    "document_pipeline.py",   C_PROC),
    (IX, 5.0,  "Embed & Store",    "vector_store.py",        C_STORE),
]
for x, y, t, s, c in idx:
    box(x, y, BW, BH, t, s, c)

lbsi = ["Path", "stats + spinners", "list[Document]", "chunks"]
for i in range(len(idx)-1):
    arrow(idx[i][0], idx[i][1]-BH/2, idx[i+1][0], idx[i+1][1]+BH/2, lbsi[i])


# ── ChromaDB (centre) ─────────────────────────────────────────────────────────
CX, CY = 7.0, 2.8
box(CX, CY, 2.8, 0.78, "ChromaDB", "persistent vector store", C_STORE)
arrow(IX, idx[-1][1]-BH/2, CX-1.4, CY+0.2, "upsert")


# ── Doc management sidebar ────────────────────────────────────────────────────
DX = 1.1
box(DX, 3.8, 1.8, 0.65, "Doc Manager", "ui.py sidebar", C_USER)
curved(DX, 3.8-0.33, CX-1.4, CY-0.1, rad=-0.3, label="list / remove")


# ── Querying pipeline ─────────────────────────────────────────────────────────
QX = 10.5
qry = [
    (QX, 9.4,  "User Question",      "ui.py",                   C_USER),
    (QX, 8.3,  "Semantic Search",    "vector_store.py",          C_STORE),
    (QX, 7.2,  "Build Context",      "vector_store.py",          C_PROC),
    (QX, 6.1,  "LLM (streaming)",    "rag.py",                   C_LLM),
    (QX, 5.0,  "Save & Display",     "session_logger.py + ui.py",C_LOG),
]
for x, y, t, s, c in qry:
    box(x, y, BW, BH, t, s, c)

lbsq = ["query str", "results", "context", "tokens + answer"]
for i in range(len(qry)-1):
    arrow(qry[i][0], qry[i][1]-BH/2, qry[i+1][0], qry[i+1][1]+BH/2, lbsq[i])

arrow(CX+1.4, CY+0.2, QX, qry[1][1]-BH/2, "query")


# ── Web search fallback ───────────────────────────────────────────────────────
WX, WY = 13.5, 7.2
box(WX, WY, 2.0, 0.65, "Web Search", "web_search.py", C_WEB)
curved(QX+BW/2, 7.2, WX-1.0, WY, rad=-0.3, label="fallback", color=C_WEB)
curved(WX-1.0, WY-0.1, QX+BW/2, 6.4, rad=-0.3, label="context", color=C_WEB)


# ── Conversation history ──────────────────────────────────────────────────────
ax.annotate("", xy=(QX, qry[3][1]+BH/2),
            xytext=(QX+BW/2+0.1, qry[4][1]),
            arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.2,
                            connectionstyle="arc3,rad=-0.5"), zorder=2)
ax.text(QX+1.8, 5.55, "history", fontsize=7, color="#999", style="italic")


# ── Provider dropdown ─────────────────────────────────────────────────────────
PX, PY = 13.5, 5.8
box(PX, PY, 2.0, 0.65, "Providers", "OpenRouter / Anthropic\n/ Ollama", C_LLM)
arrow(PX-1.0, PY, QX+BW/2, qry[3][1], "model")


# ── Legend ────────────────────────────────────────────────────────────────────
legend = [
    (C_USER,  "User Interaction"),
    (C_PROC,  "Processing"),
    (C_STORE, "Vector Store"),
    (C_LLM,   "LLM / Provider"),
    (C_LOG,   "Output / Logging"),
    (C_WEB,   "Web Search"),
]
lx, ly = 0.3, 2.3
ax.text(lx, ly + 0.5, "Legend", fontsize=9, color="#444", fontweight="bold")
for i, (c, label) in enumerate(legend):
    y = ly - i * 0.33
    ax.add_patch(FancyBboxPatch((lx, y-0.11), 0.26, 0.22,
                                boxstyle="round,pad=0.04",
                                facecolor=c, edgecolor="white", linewidth=1.5))
    ax.text(lx+0.4, y, label, va="center", fontsize=8, color="#444")


plt.tight_layout(pad=0.4)
plt.savefig("Architecture.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("Saved Architecture.png")
