from __future__ import annotations

import argparse
import faulthandler
import os
import platform
import shutil
import sys

import torch
import torch.nn.functional as F
from kernels import get_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="adarshxs/stdlib-repro")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--iters", type=int, default=100_000)
    parser.add_argument("--cache-wipe", action="store_true")
    args = parser.parse_args()

    faulthandler.enable()

    if args.cache_wipe:
        shutil.rmtree(
            os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{args.repo_id.replace('/', '--')}"
            ),
            ignore_errors=True,
        )

    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("repo:", args.repo_id)
    print("revision:", args.revision)
    print("iters:", args.iters)

    kernel = get_kernel(
        args.repo_id,
        revision=args.revision,
        backend="cpu",
        trust_remote_code=True,
    )
    x = torch.randn(1024, dtype=torch.float32)
    out, digest = kernel.relu_and_probe(x, iters=args.iters)

    torch.testing.assert_close(out, F.relu(x))
    print("digest:", digest)
    print("ok")


if __name__ == "__main__":
    main()
