"""Tests for the export scripts that do not need LiteRT: BatchNorm folding, window selection, the wire round trip,
the metrics and the model card."""
import sys
from pathlib import Path

import pytest
import torch
from model import GestureCNN, fold_batchnorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dsp"))


def trained_like_model() -> GestureCNN:
    """A model whose BatchNorm layers have non-trivial statistics and affine parameters."""
    torch.manual_seed(0)
    m = GestureCNN()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.normal_(0, 0.3)
    return m.eval()


def test_fold_batchnorm_keeps_the_output_and_removes_the_batchnorm_layers():
    m = trained_like_model()
    x = torch.rand(4, 32, 3, 32, 32) * 2 - 1
    with torch.no_grad():
        before = m(x)
        folded = fold_batchnorm(m)
        after = folded(x)
    assert not any(isinstance(mod, torch.nn.BatchNorm2d) for mod in folded.modules())
    assert torch.allclose(before, after, atol=1e-4), (before - after).abs().max()


def test_fold_batchnorm_refuses_a_model_in_train_mode():
    with pytest.raises(ValueError):
        fold_batchnorm(GestureCNN().train())


