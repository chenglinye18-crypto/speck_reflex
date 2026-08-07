from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
def test_configs_are_explicit_placeholders() -> None:
    root = Path(__file__).resolve().parents[2]
    baseline = yaml.safe_load((root / "configs/nmnist_baseline.yaml").read_text())
    robot = yaml.safe_load((root / "configs/robot_reflex.example.yaml").read_text())
    assert baseline["seed"] == 17
    assert baseline["deployment"]["dvs_input"] is False
    assert baseline["deployment"]["generate_flash_binary"] is False
    assert robot["safety"]["stm32_is_final_arbiter"] is True
    assert robot["safety"]["speck_controls_motor_pwm"] is False
