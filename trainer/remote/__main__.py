"""python -m trainer.remote serve | receive | run <config.toml>"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m trainer.remote",
                                 description="Remote training for the Mage-Flow trainer GUI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (("serve", "run GUI jobs (cache, train) on this machine"),
                       ("receive", "wait for one config from the GUI, print its path, exit")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--tunnel", choices=["cloudflared", "none"], default="cloudflared",
                       help="cloudflared: public https link, no account needed (default). none: "
                            "use your own reachable address, e.g. an exposed pod port")
        p.add_argument("--host", default=None,
                       help="bind address; default 127.0.0.1 with a tunnel, 0.0.0.0 without")
        p.add_argument("--url", default=None,
                       help="public base URL to print in the connect link (with --tunnel none)")
        p.add_argument("--token", default=None,
                       help="fixed access token (default: random; or MAGEFLOW_REMOTE_TOKEN)")
    sub.choices["receive"].add_argument("--save-dir", default=None)
    r = sub.add_parser("run", help="cache then train a config here, in the foreground")
    r.add_argument("config")
    r.add_argument("--steps", default="cache,train", help="comma list of cache, cache_dry, train")
    r.add_argument("--gpus", default="", help="device list, e.g. 0,1 (default: all)")
    r.add_argument("--num-processes", type=int, default=None)
    args = ap.parse_args()

    from . import receive_config, run, serve

    if args.cmd == "serve":
        serve(port=args.port, tunnel=args.tunnel, host=args.host, token=args.token, url=args.url)
        return 0
    if args.cmd == "receive":
        path = receive_config(port=args.port, tunnel=args.tunnel, host=args.host,
                              token=args.token, url=args.url, save_dir=args.save_dir)
        print(path)
        return 0
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    return run(args.config, steps=steps, gpus=args.gpus, num_processes=args.num_processes)


if __name__ == "__main__":
    sys.exit(main())
