"""Tests for the causal Event Observer sidecar model."""

import torch

from rlinf.models.embodiment.event_observer import (
    EventObserver,
    EventValueCritic,
    event_observer_supervision_loss,
)


def test_event_observer_is_causal_and_value_aligned() -> None:
    """Changing future features must not change an earlier event prediction."""
    torch.manual_seed(0)
    observer = EventObserver(5, 9, 8, hidden_dim=12)
    features = torch.randn(2, 4, 5)
    first = observer(features, torch.tensor([[0, 0, 1, 2], [1, 1, 1, 2]]))
    altered = features.clone()
    altered[:, 3] += 100.0
    second = observer(altered, torch.tensor([[0, 0, 1, 2], [1, 1, 1, 2]]))
    torch.testing.assert_close(first.representation[:, :3], second.representation[:, :3])
    values = EventValueCritic(12, hidden_dim=7)(first.representation)
    assert values.shape == (2, 4)


def test_event_observer_supervision_masks_missing_tracks() -> None:
    """Missing visual tracks can be excluded without changing valid losses."""
    torch.manual_seed(1)
    observer = EventObserver(3, 4, 3, hidden_dim=8)
    prediction = observer(torch.randn(1, 3, 3))
    kwargs = {
        "posterior_target": torch.zeros(1, 3, 4),
        "state_target": torch.tensor([[0, 1, 2]]),
        "boundary_target": torch.tensor([[1.0, 0.0, 1.0]]),
        "progress_target": torch.tensor([[0.0, 0.5, 1.0]]),
        "valid_mask": torch.tensor([[True, False, True]]),
    }
    losses = event_observer_supervision_loss(prediction, **kwargs)
    assert set(losses) == {"posterior", "state", "boundary", "progress"}
    assert all(bool(torch.isfinite(value)) for value in losses.values())
