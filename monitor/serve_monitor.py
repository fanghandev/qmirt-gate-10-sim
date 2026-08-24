#!/usr/bin/env python3
"""Serve a local dashboard for sparse-SRM campaign progress.

Fetches progress JSON produced by report_campaign_progress.py, either from a local
path or over SSH, caches it, and serves it to the browser. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

STATIC_DIR = Path(__file__).resolve().parent


class ProgressCache:
    def __init__(self, fetch_command: list[str] | None, local_path: Path | None):
        self.fetch_command = fetch_command
        self.local_path = local_path
        self.lock = threading.Lock()
        self.payload: dict = {"status": "starting"}
        self.fetched_at = 0.0
        self.error: str | None = None

    def refresh(self) -> None:
        try:
            if self.local_path is not None:
                text = self.local_path.read_text()
            else:
                completed = subprocess.run(
                    self.fetch_command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
                text = completed.stdout
            data = json.loads(text)
        except Exception as exc:  # noqa: BLE001 - surfaced to the dashboard
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
            return
        with self.lock:
            self.payload = data
            self.fetched_at = time.time()
            self.error = None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "fetched_at": self.fetched_at,
                "error": self.error,
                "progress": self.payload,
            }


def poll_loop(cache: ProgressCache, interval_s: float, stop: threading.Event) -> None:
    while not stop.is_set():
        cache.refresh()
        stop.wait(interval_s)


def make_handler(cache: ProgressCache):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict) -> None:
            self._send(
                code,
                json.dumps(payload, default=str).encode(),
                "application/json",
            )

        def _pixel_query(self) -> None:
            query = parse_qs(urlsplit(self.path).query)
            try:
                label = query["label"][0]
                crystal = int(query["crystal"][0])
                pixel = int(query["pixel"][0])
            except (KeyError, IndexError, ValueError):
                self._send_json(
                    400,
                    {"available": False, "error": "invalid pixel query"},
                )
                return

            with cache.lock:
                report = cache.payload
            srm = report.get("srm", {}).get(label, {})
            srm_path = srm.get("path")
            if not srm_path:
                self._send_json(
                    404,
                    {"available": False, "error": f"SRM label not found: {label}"},
                )
                return

            command = [
                sys.executable,
                str(
                    STATIC_DIR.parent
                    / "payload"
                    / "python"
                    / "report_campaign_progress.py"
                ),
                "--pixel-query",
                "--srm-dir",
                str(Path(srm_path).parent),
                "--srm-labels",
                label,
                "--crystal",
                str(crystal),
                "--pixel",
                str(pixel),
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
                payload = json.loads(completed.stdout)
            except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
                self._send_json(
                    500,
                    {"available": False, "error": f"{type(exc).__name__}: {exc}"},
                )
                return
            self._send_json(200, payload)

        def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path.startswith("/api/progress"):
                self._send_json(200, cache.snapshot())
                return
            if self.path.startswith("/api/pixel"):
                self._pixel_query()
                return
            if self.path in ("/", "/index.html"):
                index = STATIC_DIR / "index.html"
                if not index.is_file():
                    self._send(404, b"index.html missing", "text/plain")
                    return
                self._send(200, index.read_bytes(), "text/html; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local SRM progress dashboard.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--local-json",
        type=Path,
        help="Path to a progress JSON on this machine.",
    )
    source.add_argument(
        "--ssh-host",
        help="SSH destination, e.g. user@data.bridges2.psc.edu.",
    )
    source.add_argument(
        "--fetch-command",
        help="Arbitrary shell command whose stdout is the progress JSON.",
    )
    parser.add_argument(
        "--remote-json",
        help="Path to the progress JSON on the SSH host (with --ssh-host).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ssh_host and not args.remote_json:
        raise SystemExit("--ssh-host requires --remote-json")

    fetch_command = None
    if args.ssh_host:
        # -C matters: the projection payload is JSON and compresses roughly 10x.
        fetch_command = [
            "ssh",
            "-C",
            args.ssh_host,
            f"cat {shlex.quote(args.remote_json)}",
        ]
    elif args.fetch_command:
        fetch_command = shlex.split(args.fetch_command)

    cache = ProgressCache(fetch_command, args.local_json)
    stop = threading.Event()
    thread = threading.Thread(
        target=poll_loop, args=(cache, args.interval_s, stop), daemon=True
    )
    thread.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(cache))
    print(f"SRM monitor on http://{args.host}:{args.port} (poll {args.interval_s}s)")
    if fetch_command:
        print("Fetch:", " ".join(fetch_command))
    else:
        print("Fetch: local file", args.local_json)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        stop.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
