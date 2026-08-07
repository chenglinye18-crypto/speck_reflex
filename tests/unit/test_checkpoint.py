from pathlib import Path

import pytest

from speck_reflex.official_baseline.results import sha256_file


@pytest.mark.unit
def test_official_checkpoint_integrity() -> None:
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "third_party/synsense/sinabs/docs/tutorials/scnn_mnist.nir"
    assert checkpoint.is_file()
    assert sha256_file(checkpoint) == "e2fa55bda7aab5a772485e1b690358bcb825b303eca7dc426e3973937fcb5bcb"
