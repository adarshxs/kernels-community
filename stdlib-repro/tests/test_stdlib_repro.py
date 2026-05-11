import torch
import torch.nn.functional as F

import stdlib_repro


def test_relu_and_probe():
    x = torch.randn(1024, dtype=torch.float32)

    out, digest = stdlib_repro.relu_and_probe(x, iters=100)

    torch.testing.assert_close(out, F.relu(x))
    assert isinstance(digest, str)
    assert len(digest) == 32


def test_probe_stdlib():
    digest = stdlib_repro.probe_stdlib(iters=100)

    assert isinstance(digest, str)
    assert len(digest) == 32
