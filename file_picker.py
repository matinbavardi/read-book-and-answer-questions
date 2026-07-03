import json
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog
from pathlib import Path


_STATE_FILE = Path(__file__).parent / ".picker_state.json"

_FILE_FILTER = "*.pdf *.doc *.docx *.txt"


def _load_last_dir() -> Path:
    try:
        data = json.loads(_STATE_FILE.read_text())
        p = Path(data["last_dir"])
        if p.is_dir():
            return p
    except Exception:
        pass
    return Path.home()


def _save_last_dir(path: Path) -> None:
    try:
        _STATE_FILE.write_text(json.dumps({"last_dir": str(path)}))
    except Exception:
        pass


def _pick_via_zenity(title: str, initial_dir: Path) -> str | None:
    result = subprocess.run(
        [
            "zenity",
            "--file-selection",
            f"--title={title}",
            f"--filename={initial_dir}/",
            f"--file-filter=Documents | {_FILE_FILTER}",
            "--file-filter=All files | *",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _pick_via_tkinter(title: str, initial_dir: Path) -> str | None:
    supported_types = [
        ("Documents", "*.pdf *.doc *.docx *.txt"),
        ("PDF files", "*.pdf"),
        ("Word documents", "*.doc *.docx"),
        ("Text files", "*.txt"),
        ("All files", "*.*"),
    ]
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path_str = filedialog.askopenfilename(
        title=title,
        filetypes=supported_types,
        initialdir=initial_dir,
    )
    root.destroy()
    return path_str or None


def pick_file(title: str = "Select a file") -> Path | None:
    """Open a native file picker and return the selected path, or None if cancelled."""
    initial_dir = _load_last_dir()

    if shutil.which("zenity"):
        path_str = _pick_via_zenity(title, initial_dir)
    else:
        path_str = _pick_via_tkinter(title, initial_dir)

    if not path_str:
        return None

    selected = Path(path_str)
    _save_last_dir(selected.parent)
    return selected


if __name__ == "__main__":
    selected = pick_file()
    if selected:
        print(f"Selected: {selected}")
    else:
        print("No file selected.")
