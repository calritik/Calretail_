"""
Vercel entrypoint for the Dash console.

Vercel's Python runtime looks for a WSGI callable named ``app`` in this module
and routes every request to it (see vercel.json). Dash builds on Flask, so its
``.server`` is already exactly that — no adapter needed.

Only the console lives here. The FastAPI backend cannot: its dependency set
(pandas, scipy, scikit-learn, xgboost) is ~427 MB against Vercel's 250 MB
function limit, and it relies on in-process caches that a serverless invocation
throws away. It runs on Render instead, and this app reaches it through
CALRETAIL_API_BASE.
"""
import sys
from pathlib import Path

# The repo root, so `import frontend_dash...` resolves from inside api/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontend_dash.app import app as _dash_app  # noqa: E402

app = _dash_app.server
