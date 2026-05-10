# tests/test_modes.py
import pytest

from src.modes import (
    INTRINSIC_MODES, ALPHA_MODES,
    required_inputs, output_keys, validate_mode,
    InvalidModeError,
)


def test_t2RAIN_requires_no_inputs():
    assert required_inputs("t2RAIN") == set()


def test_t2RAIN_outputs_all_four():
    assert output_keys("t2RAIN") == ["rgb", "albedo", "irradiance", "normal_unit"]


def test_R2AIN_requires_rgb_only():
    assert required_inputs("R2AIN") == {"rgb"}


def test_R2AIN_outputs_three():
    assert output_keys("R2AIN") == ["albedo", "irradiance", "normal_unit"]


def test_RAI2N_requires_three():
    assert required_inputs("RAI2N") == {"rgb", "albedo", "irradiance"}


def test_RAI2N_outputs_one():
    assert output_keys("RAI2N") == ["normal_unit"]


def test_alpha_t2RPFB_outputs_four():
    assert output_keys("t2RPFB") == ["rgb", "pha", "fgr", "bgr"]


def test_alpha_R2PFB_requires_rgb():
    assert required_inputs("R2PFB") == {"rgb"}


def test_alpha_R2PFB_outputs_three():
    assert output_keys("R2PFB") == ["pha", "fgr", "bgr"]


def test_invalid_mode_raises():
    with pytest.raises(InvalidModeError):
        required_inputs("NOPE")


def test_intrinsic_mode_count_is_15():
    assert len(INTRINSIC_MODES) == 15


def test_alpha_mode_count_is_15():
    assert len(ALPHA_MODES) == 15


def test_validate_mode_passes_with_required_inputs_present():
    validate_mode("R2AIN", supplied_inputs={"rgb"})


def test_validate_mode_raises_when_required_input_missing():
    with pytest.raises(ValueError, match="rgb"):
        validate_mode("R2AIN", supplied_inputs=set())
