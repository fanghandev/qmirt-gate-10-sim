#!/usr/bin/env python3
"""Serve a local dashboard for sparse-SRM campaign progress.

Fetches progress JSON produced by report_campaign_progress.py, either from a local
path or over SSH, caches it, and serves it to the browser. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import posixpath
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
    def __init__(
        self,
        fetch_command: list[str] | None,
        local_path: Path | None,
        name: str = "default",
        label: str | None = None,
        ssh_host: str | None = None,
        remote_repo_root: str | None = None,
        root: Path | None = None,
    ):
        self.fetch_command = fetch_command
        self.local_path = local_path
        # When set, the newest batch under this root is re-resolved on every poll,
        # so a new campaign appears without restarting the service.
        self.root = root
        self.name = name
        self.label = label or name
        self.ssh_host = ssh_host
        self.remote_repo_root = remote_repo_root
        self.lock = threading.Lock()
        self.payload: dict = {"status": "starting"}
        self.fetched_at = 0.0
        self.error: str | None = None

    def resolve_path(self) -> Path | None:
        if self.root is None:
            return self.local_path
        candidates = sorted(self.root.glob("*/progress.json"))
        return candidates[-1] if candidates else None

    def refresh(self) -> None:
        try:
            if self.fetch_command is None:
                path = self.resolve_path()
                if path is None:
                    with self.lock:
                        self.error = f"no campaign report yet under {self.root}"
                    return
                text = path.read_text()
                self.local_path = path
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
                "campaign": self.name,
                "label": self.label,
                "fetched_at": self.fetched_at,
                "error": self.error,
                "progress": self.payload,
            }


def poll_loop(
    caches: list[ProgressCache], interval_s: float, stop: threading.Event
) -> None:
    while not stop.is_set():
        for cache in caches:
            cache.refresh()
        stop.wait(interval_s)


def make_handler(caches: dict[str, ProgressCache], default_name: str):
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

        def _select_cache(self, query: dict) -> ProgressCache | None:
            name = (query.get("campaign") or [default_name])[0]
            return caches.get(name)

        def _pixel_query(self) -> None:
            query = parse_qs(urlsplit(self.path).query)
            cache = self._select_cache(query)
            if cache is None:
                self._send_json(404, {"available": False, "error": "unknown campaign"})
                return
            ssh_host = cache.ssh_host
            remote_repo_root = cache.remote_repo_root
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

            if ssh_host:
                # progress.json was fetched over SSH, so srm_path is a path on the
                # remote host, not this machine — the .npz itself was never copied here.
                if not remote_repo_root:
                    self._send_json(
                        501,
                        {
                            "available": False,
                            "error": (
                                "Pixel queries need --remote-repo-root when using "
                                "--ssh-host, so report_campaign_progress.py can be run "
                                "on the remote host against its own filesystem."
                            ),
                        },
                    )
                    return
                remote_srm_dir = posixpath.dirname(srm_path)
                remote_script = posixpath.join(
                    remote_repo_root, "payload", "python", "report_campaign_progress.py"
                )
                remote_cmd = (
                    f"python3 {shlex.quote(remote_script)} --pixel-query "
                    f"--srm-dir {shlex.quote(remote_srm_dir)} "
                    f"--srm-labels {shlex.quote(label)} "
                    f"--crystal {crystal} --pixel {pixel}"
                )
                command = ["ssh", "-C", ssh_host, remote_cmd]
            else:
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
            if self.path.startswith("/api/campaigns"):
                self._send_json(
                    200,
                    {
                        "default": default_name,
                        "campaigns": [
                            {"name": cache.name, "label": cache.label}
                            for cache in caches.values()
                        ],
                    },
                )
                return
            if self.path.startswith("/api/progress"):
                query = parse_qs(urlsplit(self.path).query)
                cache = self._select_cache(query)
                if cache is None:
                    self._send_json(404, {"error": "unknown campaign"})
                    return
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
    source.add_argument(
        "--campaign",
        action="append",
        metavar="NAME=PATH",
        help=(
            "Serve several campaigns at once, e.g. "
            "--campaign 'Brain (Expanse)=/data/.../progress.json'. Repeatable; the "
            "dashboard shows a selector. Paths must be local to this machine."
        ),
    )
    source.add_argument(
        "--campaign-root",
        action="append",
        metavar="NAME=DIR",
        help=(
            "Like --campaign, but points at a directory holding batch_*/progress.json. "
            "The newest batch is re-resolved on every poll, so a new campaign is picked "
            "up without restarting. Repeatable, and can be mixed with --campaign."
        ),
    )
    parser.add_argument(
        "--remote-json",
        help="Path to the progress JSON on the SSH host (with --ssh-host).",
    )
    parser.add_argument(
        "--remote-repo-root",
        help=(
            "Path to this repo's root on the SSH host (with --ssh-host), so pixel-level "
            "queries can run report_campaign_progress.py remotely against the actual "
            ".npz files instead of failing with 'SRM not found' on this machine."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ssh_host and not args.remote_json:
        raise SystemExit("--ssh-host requires --remote-json")

    caches: dict[str, ProgressCache] = {}
    if args.campaign or args.campaign_root:
        for entry, is_root in [(item, False) for item in (args.campaign or [])] + [
            (item, True) for item in (args.campaign_root or [])
        ]:
            label, separator, raw_path = entry.partition("=")
            if not separator or not raw_path.strip():
                raise SystemExit(f"--campaign expects NAME=PATH, got: {entry!r}")
            label = label.strip()
            # The name is what travels in query strings; the label is for display.
            name = "".join(
                character if character.isalnum() else "_" for character in label
            ).strip("_")
            if name in caches:
                raise SystemExit(f"duplicate campaign name: {label}")
            target = Path(raw_path.strip()).expanduser()
            caches[name] = ProgressCache(
                None,
                None if is_root else target,
                name=name,
                label=label,
                remote_repo_root=args.remote_repo_root,
                root=target if is_root else None,
            )
    else:
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
        caches["default"] = ProgressCache(
            fetch_command,
            args.local_json,
            name="default",
            label="campaign",
            ssh_host=args.ssh_host,
            remote_repo_root=args.remote_repo_root,
        )

    default_name = next(iter(caches))
    stop = threading.Event()
    thread = threading.Thread(
        target=poll_loop,
        args=(list(caches.values()), args.interval_s, stop),
        daemon=True,
    )
    thread.start()

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(caches, default_name)
    )
    print(f"SRM monitor on http://{args.host}:{args.port} (poll {args.interval_s}s)")
    for cache in caches.values():
        if cache.fetch_command:
            print(f"  {cache.label}: {' '.join(cache.fetch_command)}")
        elif cache.root is not None:
            print(f"  {cache.label}: newest batch under {cache.root}")
        else:
            print(f"  {cache.label}: local file {cache.local_path}")
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
