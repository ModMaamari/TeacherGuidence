"""
Stdlib HTTP server for the trajectory explorer.

Serves the vanilla frontend (``static/``) and a small JSON API backed by the
data-access layer. No third-party dependencies.

Routes:
    GET /                                serves the explorer page
    GET /static/<file>                   static assets
    GET /api/runs                        list runs + aggregate metrics
    GET /api/episodes?run=<run_id>       per-episode summaries for a run
    GET /api/episode?run=<run_id>&qid=   full episode
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

from agentsim.teacher_guidance.viewer import data_access as da

STATIC_DIR = Path(__file__).parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "TeacherGuidanceViewer/1.0"

    # --- helpers -----------------------------------------------------------
    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel_path: str) -> None:
        # Constrain to STATIC_DIR (reject traversal).
        target = (STATIC_DIR / rel_path).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_json({"error": "forbidden"}, 403)
            return
        if not target.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        content_type = _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self._send_bytes(target.read_bytes(), content_type)

    @property
    def output_root(self) -> str:
        return self.server.output_root  # type: ignore[attr-defined]

    # --- routing -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._serve_static("index.html")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        elif path == "/api/runs":
            self._send_json(da.find_runs(self.output_root))
        elif path == "/api/episodes":
            run = (params.get("run") or [""])[0]
            self._send_json(da.get_run_episodes(self.output_root, run))
        elif path == "/api/episode":
            run = (params.get("run") or [""])[0]
            qid = (params.get("qid") or [""])[0]
            episode = da.get_episode(self.output_root, run, qid)
            if episode is None:
                self._send_json({"error": "episode not found"}, 404)
            else:
                self._send_json(episode)
        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, *args: Any) -> None:  # silence default stderr logging
        pass


def make_server(output_root: str, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), ViewerHandler)
    httpd.output_root = str(output_root)  # type: ignore[attr-defined]
    return httpd


def serve(
    output_root: str = "data/simulation_output",
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    httpd = make_server(output_root, host, port)
    url = f"http://{host}:{port}/"
    print(f"Teacher Guidance trajectory explorer running at {url}")
    print(f"Serving runs from: {Path(output_root).resolve()}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        httpd.server_close()
