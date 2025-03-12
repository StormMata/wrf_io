import os
import pytest
from wrf_io import preproc

@pytest.fixture
def namelist_path():
    """Return the path to a real namelist.input file in the tests directory."""
    return os.path.join(os.path.dirname(__file__), "namelist.input")

def turbine_properties_path():
    """Return the path to a real turbineProperties.tbl file in the tests directory."""
    return os.path.join(os.path.dirname(__file__), "turbineProperties.tbl")

def test_parse_namelist(namelist_path):
    """Test parse_namelist with a real namelist.input file."""
    opt_params = {"name_path": namelist_path}
    config = preproc.parse_namelist(opt_params)

    assert isinstance(config, dict)

    # Sample checks based on expected values in your namelist.input
    assert "time_control" in config
    assert "domains" in config
    assert "physics" in config
    assert "dynamics" in config
    assert "bdy_control" in config

    # Check a few expected values (update these based on your actual namelist)
    # assert config["time_control"]["run_days"] >= 0
    # assert config["domains"]["time_step"] > 0

def test_parse_turbine_properties():
    """Test parse_turbine_properties with a real turbineProperties.tbl file."""
    opt_params = {
        "read_from": os.path.dirname('./tests/'),
        "turb_model": "iea10MW"  # Ensure your test file is placed correctly
    }

    config = preproc.parse_turbine_properties(opt_params)

    assert isinstance(config, dict)
    assert len(config) > 0, "Parsed configuration is empty"

    # List of expected keys
    expected_keys = [
        "Length of the blade element vector [-]",
        "Number of airfoils [-]",
        "Number of blades [-]",
        "Hub height [m]",
        "Tower diameter [m]",
        "Rotor diameter [m]",
        "Hub diameter [m]",
        "Nacelle length [m]",
        "Nacelle drag coefficient [-]",
        "Tower drag coefficient [-]",
        "OverHang [m]",
        "UndSling [m]",
        "Electrical efficiency of the generator [-]",
        "Delta (Turbine tilt) [deg]",
        "Zeta (Blade precone) [deg]",
        "Rotation direction [-], 1: clockwise, -1: counter-clockwise, 0: no rotational effects (GAD/GADrs; only for experimental use)",
        "CT, Constant thrust coefficient for GAD/GADrs. It should be applied when rotational effects are neglected. E.g., CT=0.77 for Uinf=8.0 m s-1. Only for experimental use.",
        "blade_c_eps_1 [-], x or chord-wise",
        "blade_c_eps_2 [-], y or thickness-wise",
        "blade_c_eps_3 [-], z or radial-wise",
        "nacelle_c_eps [-]",
        "tower_c_eps [-]",
        "Inflow location [m]",
        "Yaw rate [deg/s]",
        "Yaw error threshold [(deg)^2*(seconds)]"
    ]

    for key in expected_keys:
        assert key in config, f"Missing expected key: {key}"

    # Check that all values are of type float
    for key, value in config.items():
        assert isinstance(value, float), f"Value for {key} is not a float"