#!/usr/bin/env python
"""Launch the web front-end (photo / video / live-camera) for InferencePipeline.

    python scripts/run_web.py
    python scripts/run_web.py --port 8080
    python scripts/run_web.py --pipeline-config configs/pipeline.yaml

Then open the printed URL in a browser. The pipeline (detector + OCR
models) loads once at startup — see the log line "pipeline ready" — the
first request after that is already fast; there's no per-request reload.

Uses uvicorn's factory mode (`plaka.web.app:create_app`, factory=True) so
the real InferencePipeline is only built when this script actually runs
the server, not just by importing plaka.web.app (tests import create_app()
directly with a fake pipeline — see tests/unit/test_web_app.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from plaka.pipeline.builder import REPO_ROOT

DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
    args = parser.parse_args()

    # uvicorn's factory mode only accepts an import string, so the config
    # path has to reach create_app() through an env var rather than a
    # normal function argument.
    import os

    os.environ["PLAKA_PIPELINE_CONFIG"] = str(args.pipeline_config)

    uvicorn.run(
        "plaka.web.app:create_app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
