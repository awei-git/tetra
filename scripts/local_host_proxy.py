"""Simple local reverse proxy for mapping hostnames to local ports."""

from __future__ import annotations

import argparse
import http.client
import http.server
import socketserver
from pathlib import Path
from typing import Dict, Optional

DEFAULT_LISTEN_PORT = 8080
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def load_host_map(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not path.exists():
        return mapping
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "=" not in raw:
            continue
        host, port = raw.split("=", 1)
        host = host.strip().lower()
        port = port.strip()
        if not host:
            continue
        try:
            mapping[host] = int(port)
        except ValueError:
            continue
    return mapping


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _handle(self) -> None:
        host_header = self.headers.get("Host", "")
        host = host_header.split(":", 1)[0].lower()
        port = self.server.host_map.get(host)
        if not port:
            self.send_error(404, f"Unknown host: {host or 'missing'}")
            return

        content_length = self.headers.get("Content-Length")
        body = None
        if content_length:
            try:
                body = self.rfile.read(int(content_length))
            except ValueError:
                body = None

        target_host = "127.0.0.1"
        conn = http.client.HTTPConnection(target_host, port, timeout=30)
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_HEADERS and key.lower() != "host"
        }
        headers["Host"] = f"{target_host}:{port}"
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
        except Exception as exc:
            self.send_error(502, f"Upstream error: {exc}")
            return
        finally:
            conn.close()

        self.send_response(resp.status, resp.reason)
        for key, value in resp.getheaders():
            if key.lower() in HOP_HEADERS:
                continue
            if key.lower() == "content-length":
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()

    def do_PATCH(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle()

    def log_message(self, format: str, *args: object) -> None:
        return


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, host_map: Dict[str, int]):
        super().__init__(server_address, RequestHandlerClass)
        self.host_map = host_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local hostname reverse proxy")
    parser.add_argument("--port", type=int, default=DEFAULT_LISTEN_PORT, help="Port to listen on")
    parser.add_argument("--map", dest="map_path", type=str, help="Path to host map file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    map_path = Path(args.map_path) if args.map_path else base_dir / "config" / "hostmap.txt"
    host_map = load_host_map(map_path)
    if not host_map:
        raise SystemExit(f"No host mappings found in {map_path}")

    server = ThreadedHTTPServer(("127.0.0.1", args.port), ProxyHandler, host_map)
    print(f"Proxy listening on 127.0.0.1:{args.port} for {', '.join(sorted(host_map))}")
    server.serve_forever()


if __name__ == "__main__":
    main()
