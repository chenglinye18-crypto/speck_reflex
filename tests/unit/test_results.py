import pytest

from speck_reflex.official_baseline.results import validate_results


@pytest.mark.unit
def test_minimal_result_schema() -> None:
    data = {key: {} for key in ("sample", "sinabs_software", "dynapcnn", "samna_config", "specksim", "hardware")}
    data.update(mode="pipeline_smoke", seed=17)
    validate_results(data)


@pytest.mark.unit
def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError):
        validate_results({"mode": "accuracy_magic"})
