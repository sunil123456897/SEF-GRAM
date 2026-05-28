from pathlib import Path
import sys


# The repository currently keeps the runnable project inside the SEF-GRAM/
# subdirectory. When pytest is launched from different working directories,
# Python may not automatically put that package root on sys.path. Keep imports
# stable for both layouts:
#   repo_root/SEF-GRAM/sef_gram
#   repo_root/sef_gram
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _PROJECT_ROOT

for path in (_PROJECT_ROOT, _REPO_ROOT, _REPO_ROOT / "SEF-GRAM"):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
