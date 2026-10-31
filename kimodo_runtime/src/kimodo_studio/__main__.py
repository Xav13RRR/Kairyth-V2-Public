# SPDX-FileCopyrightText: Copyright (c) 2026 MKA Project contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kimodo_studio.config import SHUTDOWN_GRACE_S, STUDIO_HOST, STUDIO_PORT
from kimodo_studio.lifecycle import request_shutdown, write_studio_pid
from kimodo_studio.server import app, init_session


def main() -> int:
    parser = argparse.ArgumentParser(description="Kimodo Studio — independent animation UI")
    parser.add_argument(
        "--binding",
        type=str,
        default=os.environ.get("KIMODO_PREPARED_BINDING", ""),
        help="Path to prepared_binding.npz",
    )
    parser.add_argument("--port", type=int, default=STUDIO_PORT)
    parser.add_argument("--host", type=str, default=STUDIO_HOST)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.binding:
        print("Error: --binding or KIMODO_PREPARED_BINDING required", file=sys.stderr)
        return 1

    binding_path = Path(args.binding).resolve()
    try:
        init_session(binding_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}/"
    print(f"[Kimodo Studio] binding={binding_path}")
    print(f"[Kimodo Studio] UI: {url}")
    print(f"[Kimodo Studio] 关闭浏览器标签页后 {SHUTDOWN_GRACE_S:g}s 内自动退出并清理后台进程")

    if not args.no_browser:
        webbrowser.open(url)

    import uvicorn

    write_studio_pid()
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)

    def _watch_shutdown() -> None:
        from kimodo_studio.lifecycle import shutdown_event

        while not shutdown_event().is_set():
            time.sleep(0.2)
        server.should_exit = True

    threading.Thread(target=_watch_shutdown, name="kimodo-studio-shutdown", daemon=True).start()

    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[Kimodo Studio] Interrupted")
    finally:
        request_shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
