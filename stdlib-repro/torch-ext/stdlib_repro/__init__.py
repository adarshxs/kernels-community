from __future__ import annotations

from typing import Optional

import torch

from ._ops import ops

DEFAULT_CODE = """\
#include <deep_gemm/impls/smxx_layout.cuh>
#include <deep_gemm/common/compile.cuh>
#include <cutlass/detail/helper_macros.hpp>
#include "local_header.cuh"
"""


def relu_and_probe(
    x: torch.Tensor,
    *,
    code: str = DEFAULT_CODE,
    iters: int = 1000,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, str]:
    if out is None:
        out = torch.empty_like(x)
    digest = ops.relu_and_probe(out, x, code, iters)
    return out, digest


def probe_stdlib(*, code: str = DEFAULT_CODE, iters: int = 1000) -> str:
    x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float32)
    _, digest = relu_and_probe(x, code=code, iters=iters)
    return digest


__all__ = ["DEFAULT_CODE", "probe_stdlib", "relu_and_probe"]
