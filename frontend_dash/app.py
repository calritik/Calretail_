"""
CalRetail — Dash Retail AI Intelligence Console.

Implements a curated set of capability domains from the Calsoft Retail AI deck
in the Industrial-AI portfolio-console design language.
"""
import os
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import dash
from dash import Dash, Input, Output, dcc, html

from frontend_dash.components.layout import app_shell

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=str(Path(__file__).parent / "pages"),
    suppress_callback_exceptions=True,
    title="CalRetail AI",
    update_title=None,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
    ],
)
server = app.server

# The theme class is set before first paint, ahead of the Dash bundle, so a
# dark-mode reload never flashes a white page. assets/theme.js owns it after
# that; keep the storage key in sync between the two.
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>
          (function () {
            try {
              if (localStorage.getItem('calretail-theme') === 'dark') {
                document.documentElement.setAttribute('data-boot-theme', 'dark');
              }
            } catch (e) {}
          })();
        </script>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>"""

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="shell"),
    ]
)


@app.callback(Output("shell", "children"), Input("url", "pathname"))
def _render_shell(pathname):
    return app_shell(pathname or "/", dash.page_container)


if __name__ == "__main__":
    # debug=False deliberately: Dash's dev-tools widget anchors bottom-right and
    # sits on top of the last card in the grid. Set DASH_DEBUG=1 when you want
    # the callback graph and hot reload back.
    debug = os.getenv("DASH_DEBUG") == "1"
    # Local dev keeps the old 127.0.0.1:8050 defaults untouched; the container
    # start script sets DASH_HOST=0.0.0.0 and DASH_PORT to whatever the host
    # platform actually routes traffic to.
    #
    # PORT is honoured as a fallback because most container hosts (Render, Fly,
    # Cloud Run, Heroku) inject the routed port under that name and route to
    # nothing else. Binding 8050 there means the platform's health check never
    # connects and the deploy is marked failed with the app running fine.
    host = os.getenv("DASH_HOST", "127.0.0.1")
    port = int(os.getenv("DASH_PORT") or os.getenv("PORT") or "8050")
    app.run(debug=debug, use_reloader=debug, threaded=True, port=port, host=host)
