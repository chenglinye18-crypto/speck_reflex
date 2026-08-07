from pathlib import Path

import pytest
import torch

from speck_reflex.official_baseline.data import dataset_available, load_sample
from speck_reflex.official_baseline.dynapcnn import build_dynapcnn, run_specksim
from speck_reflex.official_baseline.model import build_random_quick_start_model, reset_states, set_seed


@pytest.mark.integration
def test_one_real_sample_full_software_chain() -> None:
    root = Path(__file__).resolve().parents[2]
    data_root = root / "data/official/NMNIST"
    if not dataset_available(data_root):
        pytest.skip("N-MNIST test data not present; run demo-smoke to download it")
    set_seed()
    raw, frames, label = load_sample(data_root, 0)
    model = build_random_quick_start_model().eval()
    with torch.no_grad():
        output = model(torch.as_tensor(frames).float())
    assert output.shape == (100, 10)
    dynapcnn, config, valid, _ = build_dynapcnn(model, device_type="speck2fdevkit")
    assert valid
    assert config is not None
    reset_states(dynapcnn)
    simulated = run_specksim(dynapcnn, raw)
    assert simulated.dtype.names == ("x", "y", "t", "p")
    assert label == 0
