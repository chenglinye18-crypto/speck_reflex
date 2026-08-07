import os

import pytest


@pytest.mark.hardware
def test_hardware_suite_is_explicitly_gated() -> None:
    if os.environ.get("SPECK_ALLOW_HARDWARE_TESTS") != "1":
        pytest.skip("set SPECK_ALLOW_HARDWARE_TESTS=1 only after explicit approval")
    pytest.skip("physical Speck validation is not implemented in this no-hardware phase")
