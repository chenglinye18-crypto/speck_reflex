"""Small no-training smoke run for SNN Motion Backbone v0.1."""

from __future__ import annotations

import torch

from .motion_backbone import SNNMotionBackbone


def main() -> None:
    torch.manual_seed(17)
    model = SNNMotionBackbone()
    event_bins = (torch.rand(1, 4, 2, 96, 128) < 0.01).float()
    with torch.no_grad():
        output = model(event_bins)

    print("SNN_MOTION_BACKBONE_SMOKE")
    print("mode=architecture_smoke_random_weights")
    print(f"parameters={model.parameter_count()}")
    print(f"primitive_spikes={tuple(output.primitive_spikes.shape)}")
    print(f"local_logits={tuple(output.local_logits.shape)}")
    print(f"global_embedding={tuple(output.global_embedding.shape)}")
    print(f"ego_motion={tuple(output.ego_motion.shape) if output.ego_motion is not None else None}")
    model.reset_state()
    print(f"state_reset={len(model.membrane_states()) == 0}")


if __name__ == "__main__":
    main()
