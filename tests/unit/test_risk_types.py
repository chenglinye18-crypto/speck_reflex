import pytest

from speck_reflex.runtime import RiskDirection, RiskLevel, RiskMessage


@pytest.mark.unit
def test_risk_enums_and_validation() -> None:
    assert RiskLevel.EMERGENCY_STOP == 4
    message = RiskMessage(1, 2, 3, RiskLevel.CAUTION, RiskDirection.FRONT, 0.75, 1.2, True)
    assert message.direction is RiskDirection.FRONT
    with pytest.raises(ValueError):
        RiskMessage(1, 2, 3, RiskLevel.NORMAL, RiskDirection.UNKNOWN, 1.1, None, True)
