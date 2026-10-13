import pytest

from vercausa.detection import (
    check_hdf,
    check_osf,
    check_pwf,
    check_rnf,
    check_twf,
    evaluate_reading,
    rpm_to_rad_per_s,
)


def test_rpm_conversion():
    assert rpm_to_rad_per_s(60) == pytest.approx(2 * 3.141592653589793)


def test_twf_window_boundaries():
    assert not check_twf(199)
    assert check_twf(200)
    assert check_twf(240)
    assert not check_twf(241)


def test_hdf_needs_both_conditions():
    assert check_hdf(air_temp_k=300, process_temp_k=305, rpm=1000)
    assert not check_hdf(air_temp_k=300, process_temp_k=310, rpm=1000)
    assert not check_hdf(air_temp_k=300, process_temp_k=305, rpm=1500)


def test_pwf_out_of_range_either_side():
    low_rpm = 500
    assert check_pwf(torque_nm=1.0, rpm=low_rpm)
    high_power_torque = 200.0
    assert check_pwf(torque_nm=high_power_torque, rpm=2000)
    assert not check_pwf(torque_nm=40.0, rpm=1500)


def test_osf_threshold_varies_by_variant():
    wear, torque = 200, 60
    assert wear * torque == 12000
    assert check_osf(wear, torque, "L")
    assert not check_osf(wear, torque, "M")
    assert not check_osf(wear, torque, "H")


def test_rnf_is_never_observable_from_state():
    assert check_rnf() is False


def test_evaluate_reading_collects_reasons():
    result = evaluate_reading(
        product_variant="L",
        air_temp_k=298.0,
        process_temp_k=305.0,
        rpm=1200,
        torque_nm=40.0,
        tool_wear_min=210,
    )
    assert result.triggered
    assert result.twf_candidate
    assert result.hdf
    assert result.tripped == ["HDF", "TWF"]


def test_evaluate_reading_healthy_machine_does_not_trigger():
    result = evaluate_reading(
        product_variant="M",
        air_temp_k=298.1,
        process_temp_k=308.6,
        rpm=1551,
        torque_nm=42.8,
        tool_wear_min=0,
    )
    assert not result.triggered
    assert result.reasons == {}
