from __future__ import annotations

import pytest
import torch

from software.models.snn import SNNMotionBackbone, SNNMotionConfig
from software.training.train_evimo2_ego_motion import (
    DeterministicEpochSampler,
    window_prediction,
)


@pytest.mark.unit
def test_window_readout_matches_mean_of_shared_per_timestep_head() -> None:
    torch.manual_seed(17)
    model = SNNMotionBackbone(SNNMotionConfig())
    events = (torch.rand(1, 4, 2, 16, 16) < 0.02).float()
    model.reset_state()
    output = model(events)
    assert output.ego_motion is not None
    expected = output.ego_motion.mean(dim=1)
    pooled = output.global_embedding.mean(dim=1)
    direct = model.ego_motion_head(pooled)
    assert torch.allclose(expected, direct, atol=1e-7, rtol=1e-6)

    model.reset_state()
    actual = window_prediction(model, events)
    assert actual.shape == (1, 6)
    assert torch.allclose(actual, expected)


@pytest.mark.unit
def test_epoch_sampler_is_resume_stable_and_epoch_specific() -> None:
    uninterrupted = DeterministicEpochSampler(20, seed=17)
    uninterrupted.set_epoch(2)
    expected_epoch_two = list(uninterrupted)

    resumed = DeterministicEpochSampler(20, seed=17)
    resumed.set_epoch(2)
    assert list(resumed) == expected_epoch_two

    resumed.set_epoch(1)
    assert list(resumed) != expected_epoch_two
    assert sorted(expected_epoch_two) == list(range(20))
