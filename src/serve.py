"""Yerel sunucu: docs/ klasörünü yayınlar ve sitedeki "Yeniden Analiz" butonu için
POST /api/refresh (boru hattını çalıştırır) ve GET /api/status uç noktalarını sağlar.

Kullanım:  python src/serve.py [port]     ->  http://localhost:8000
Sunucu yalnızca 127.0.0.1'e bağlanır.
"""
import json
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = {"state": "idle", "last_line": ""}
LOCK = threading.Lock()


def run_pipeline():
    try:
        p = subprocess.Popen([sys.executable, "-X", "utf8", str(ROOT / "src" / "run_pipeline.py")], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        for line in p.stdout:
            STATE["last_line"] = line.strip()[:160]
        p.wait()
        STATE["state"] = "idle" if p.returncode == 0 else "error"
    except Exception as e:  # noqa: BLE001
        STATE.update(state="error", last_line=str(e))


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            return self._json(STATE)
        super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/refresh"):
            with LOCK:
                if STATE["state"] != "running":
                    STATE.update(state="running", last_line="başlatılıyor")
                    threading.Thread(target=run_pipeline, daemon=True).start()
            return self._json(STATE, 202)
        self._json({"error": "bulunamadı"}, 404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT / "docs")))
    print(f"ZİNCİR-TR yerel sunucu: http://localhost:{port}  (durdurmak için Ctrl+C)")
    srv.serve_forever()
