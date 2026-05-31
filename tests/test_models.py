"""Contract + behavioral tests for the ``ninav.models`` position models.

These tests exercise the shared model contract enforced by the factory
:func:`ninav.models.build_model` and a few model-specific behavioral guards
that protect against historical bugs:

* the model contract: ``forward`` of a channel-first ``(B, in_channels, L)``
  batch returns ``(B, out_dim)`` -- and for the probabilistic ``tlio`` model a
  ``(mean, log_std)`` tuple, each ``(B, out_dim)``;
* a Transformer PERMUTATION test proving the sinusoidal positional encoding is
  in effect (guards bug C13: a positionless self-attention encoder is
  permutation-equivariant over time, so its mean-pooled output would be
  invariant to time shuffling);
* a Transformer head/width invariant: ``d_model % nhead == 0`` and ``nhead >= 4``;
* an LSTM TEMPORAL-ORDERING test proving the recurrent sequence length is
  ``L = 200`` and not ``1`` (guards bug C11): if the window were collapsed to a
  single timestep, reordering / reversing time would leave the output
  unchanged;
* a shape-discipline test: no model silently accepts a transposed
  ``(B, L, in_channels)`` batch as if it were the channel-first contract input
  -- it must raise.

All tests run on CPU with tiny tensors and are deterministic (seeded, models in
``eval`` mode so dropout / BatchNorm running-stats do not perturb outputs).
"""
from __future__ import annotations

import warnings

import pytest
import torch

from ninav.models import MODEL_REGISTRY, build_model
from ninav.models.transformer import TransformerRegressor


# All registered position models. ``tlio`` is the probabilistic exception whose
# forward returns a ``(mean, log_std)`` tuple instead of a single tensor.
ALL_MODELS = sorted(MODEL_REGISTRY)  # ['lstm', 'resnet1d', 'tcn', 'tlio', 'transformer']
TUPLE_MODELS = {"tlio"}

# Contract dimensions used throughout.
B, C, L, OUT = 3, 6, 200, 2


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic per-test seeding."""
    torch.manual_seed(0)


def _silence_nested_tensor_warning():
    """The Transformer encoder emits a benign ``enable_nested_tensor`` warning
    under ``norm_first=True``; suppress it so test output stays clean."""
    warnings.filterwarnings(
        "ignore",
        message=".*enable_nested_tensor.*",
        category=UserWarning,
    )


def _make(name: str) -> torch.nn.Module:
    """Build a model in eval mode (deterministic: no dropout / BN updates)."""
    _silence_nested_tensor_warning()
    model = build_model(name, in_channels=C, out_dim=OUT)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Shared contract: forward shape.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_MODELS)
def test_forward_output_contract(name):
    """``forward(randn(B, C, L))`` returns the contract output shape.

    Position models return ``(B, out_dim)``. ``tlio`` returns a
    ``(mean, log_std)`` tuple, each ``(B, out_dim)``, both finite.
    """
    model = _make(name)
    x = torch.randn(B, C, L)
    with torch.no_grad():
        out = model(x)

    if name in TUPLE_MODELS:
        assert isinstance(out, tuple), f"{name}: expected a (mean, log_std) tuple"
        assert len(out) == 2, f"{name}: expected 2 outputs, got {len(out)}"
        mean, log_std = out
        assert mean.shape == (B, OUT), f"{name}: mean shape {tuple(mean.shape)}"
        assert log_std.shape == (B, OUT), f"{name}: log_std shape {tuple(log_std.shape)}"
        assert torch.isfinite(mean).all(), f"{name}: non-finite mean"
        assert torch.isfinite(log_std).all(), f"{name}: non-finite log_std"
    else:
        assert isinstance(out, torch.Tensor), f"{name}: expected a single tensor"
        assert out.shape == (B, OUT), f"{name}: output shape {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{name}: non-finite output"


def test_tlio_returns_mean_logstd_tuple_explicitly():
    """``tlio`` specifically yields a (mean, log_std) tuple of two (B, out_dim)
    tensors -- the probabilistic contract paired with ``gaussian_nll_loss``."""
    model = _make("tlio")
    x = torch.randn(B, C, L)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, tuple) and len(out) == 2
    mean, log_std = out
    assert mean.shape == (B, OUT)
    assert log_std.shape == (B, OUT)


# --------------------------------------------------------------------------- #
# Shape discipline: no model silently accepts (B, L, C) as the contract input.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_MODELS)
def test_rejects_transposed_input(name):
    """A wrongly-shaped ``(B, L, in_channels)`` batch must NOT be accepted as if
    it were the channel-first ``(B, in_channels, L)`` contract input.

    The model must either raise, or (defensively) refuse to produce the
    contract ``(B, out_dim)`` shape. ``L != in_channels`` here (200 != 6) so any
    model that genuinely reads channels from dim 1 cannot silently succeed.
    """
    model = _make(name)
    wrong = torch.randn(B, L, C)  # transposed: (B, 200, 6) instead of (B, 6, 200)

    raised = False
    out = None
    try:
        with torch.no_grad():
            out = model(wrong)
    except (RuntimeError, ValueError) as exc:
        raised = True
        _ = exc  # any shape/channel error is acceptable

    if not raised:
        # If it did not raise, it must at least NOT have produced a result that
        # masquerades as a correct contract output.
        produced = out[0] if isinstance(out, tuple) else out
        assert produced.shape != (B, OUT), (
            f"{name}: silently accepted transposed (B, L, C) input and produced "
            f"a contract-shaped {tuple(produced.shape)} output -- shape discipline "
            "is not enforced"
        )


def test_transformer_rejects_wrong_channel_count_explicitly():
    """The Transformer explicitly validates dims and channel count and raises a
    clear ``ValueError`` (not a deep, opaque error) for bad input shapes."""
    model = _make("transformer")
    with pytest.raises(ValueError):
        with torch.no_grad():
            model(torch.randn(B, L, C))  # transposed -> wrong channel count
    with pytest.raises(ValueError):
        with torch.no_grad():
            model(torch.randn(B, C, L, 1))  # not 3D


# --------------------------------------------------------------------------- #
# Transformer: positional-encoding permutation test (guards bug C13).
# --------------------------------------------------------------------------- #
def test_transformer_is_sensitive_to_time_permutation():
    """Shuffling the time axis must change the Transformer's output by a
    non-negligible amount.

    A self-attention encoder with mean pooling is permutation-EQUIVARIANT over
    the sequence axis; mean pooling then makes the whole stack permutation-
    INVARIANT. The sinusoidal positional encoding is what breaks that symmetry.
    If positional encoding were dropped (bug C13), the time-shuffled output
    would be (numerically) identical -- so a clearly non-zero difference proves
    positional information is in effect.
    """
    model = _make("transformer")
    x = torch.randn(B, C, L)
    with torch.no_grad():
        y = model(x)
        perm = torch.randperm(L)
        # Guard against the degenerate identity permutation.
        while torch.equal(perm, torch.arange(L)):
            perm = torch.randperm(L)
        y_shuffled = model(x[:, :, perm])

    max_abs_diff = (y - y_shuffled).abs().max().item()
    assert max_abs_diff > 1e-4, (
        "time-shuffled input produced a (near) identical Transformer output "
        f"(max abs diff {max_abs_diff:.3e}); positional encoding appears to be "
        "missing or ineffective (bug C13)"
    )


def test_transformer_head_width_invariants():
    """``d_model % nhead == 0`` and ``nhead >= 4`` for the default Transformer,
    and the constructor enforces both as assertions."""
    model = _make("transformer")
    nhead = model.encoder.layers[0].self_attn.num_heads
    assert model.d_model % nhead == 0, (
        f"d_model ({model.d_model}) not divisible by nhead ({nhead})"
    )
    assert nhead >= 4, f"nhead must be >= 4, got {nhead}"

    # The constructor must reject violating configurations.
    with pytest.raises(AssertionError):
        TransformerRegressor(d_model=128, nhead=3)  # not divisible
    with pytest.raises(AssertionError):
        TransformerRegressor(d_model=128, nhead=2)  # nhead < 4


# --------------------------------------------------------------------------- #
# LSTM: temporal-ordering test (guards bug C11 -- seq dim must be L, not 1).
# --------------------------------------------------------------------------- #
def test_lstm_uses_full_temporal_sequence():
    """The LSTM must consume a length-``L`` sequence, not a length-1 one.

    If the window were fed to the recurrent layer with sequence length 1 (the
    legacy bug C11: channels read as the time axis, or time collapsed), the
    output would be invariant to the temporal ORDER of the samples. We probe
    this two ways -- reversing time and randomly permuting time -- and require
    the output to change in both cases. A length-1 (order-insensitive) model
    cannot pass.
    """
    model = _make("lstm")
    x = torch.randn(B, C, L)
    with torch.no_grad():
        y = model(x)
        # 1) Reverse the time axis.
        y_rev = model(torch.flip(x, dims=[2]))
        # 2) Random permutation of the time axis.
        perm = torch.randperm(L)
        while torch.equal(perm, torch.arange(L)):
            perm = torch.randperm(L)
        y_perm = model(x[:, :, perm])

    rev_diff = (y - y_rev).abs().max().item()
    perm_diff = (y - y_perm).abs().max().item()
    assert rev_diff > 1e-4, (
        "reversing the time axis left the LSTM output unchanged "
        f"(max abs diff {rev_diff:.3e}); the recurrent sequence length looks "
        "like 1 rather than L (bug C11)"
    )
    assert perm_diff > 1e-4, (
        "permuting the time axis left the LSTM output unchanged "
        f"(max abs diff {perm_diff:.3e}); the recurrent sequence length looks "
        "like 1 rather than L (bug C11)"
    )


def test_lstm_recurrent_sequence_length_is_L():
    """Direct structural check: the LSTM's input_size equals ``in_channels``
    (so time, length ``L``, is the sequence axis), not ``L`` (which would mean
    the channel axis was being fed as a single recurrent step)."""
    model = _make("lstm")
    assert model.lstm.input_size == C, (
        f"LSTM input_size is {model.lstm.input_size}; expected in_channels={C}. "
        "If it were L the time axis would have been collapsed to length 1 (C11)."
    )
    assert model.lstm.batch_first is True


# --------------------------------------------------------------------------- #
# Factory hygiene.
# --------------------------------------------------------------------------- #
def test_build_model_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        build_model("does_not_exist")


def test_build_model_covers_full_registry():
    """The set we test matches the documented factory registry exactly."""
    assert set(ALL_MODELS) == {"resnet1d", "transformer", "lstm", "tcn", "tlio"}
