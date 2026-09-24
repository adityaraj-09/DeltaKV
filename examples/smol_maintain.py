"""Run weight-aware cache maintenance on SmolLM2-135M.

Prefills a prompt under W0, publishes a rank-r LoRA on q/k/v without
flushing the prefix, patches the stored KV, and scores it against a
fresh prefill under W1.
"""

from __future__ import annotations

import sys

from deltakv.cli import main


if __name__ == "__main__":
    argv = [
        "maintain",
        "--model",
        "HuggingFaceTB/SmolLM2-135M",
        "--prompt",
        "France is a country in Western Europe. It is known for wine, cheese, and the city of lights. Many students learn that Paris sits on the Seine. The capital of France is",
        "--seq",
        "32",
        "--rank",
        "4",
        "--scale",
        "0.01",
        "--route",
        "hybrid",
    ]
    if len(sys.argv) > 1:
        argv.extend(sys.argv[1:])
    raise SystemExit(main(argv))
