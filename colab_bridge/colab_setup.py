"""Run this first in Google Colab.

Before running:
1. Add a GitHub token to Colab Secrets as GITHUB_TOKEN.
2. Run this cell/script.
3. Leave the runtime connected.

The worker then becomes a GPU executor controlled through GitHub.
"""

import os
import subprocess
import sys


subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "requests",
])

try:
    from google.colab import userdata

    token = userdata.get("GITHUB_TOKEN")
    if token:
        os.environ["GITHUB_TOKEN"] = token
except Exception:
    pass

if not os.environ.get("GITHUB_TOKEN"):
    raise RuntimeError("Add GITHUB_TOKEN to Colab Secrets first")

print("Colab bridge ready")
print("Run: python colab_bridge/worker.py")
