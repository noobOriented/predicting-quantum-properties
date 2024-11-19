import pathlib

import pytest
from typer.testing import CliRunner

import data_acquisition_shadow
import generate_observables
import prediction_shadow


runner = CliRunner()

SNAPSHOT_PATH = pathlib.Path(__file__).parent / 'snapshots'


def test_prediction_shadow():
    result = runner.invoke(prediction_shadow.app, ['measurement.txt', 'observables.txt'])

    assert result.exit_code == 0

    expected_result = open(SNAPSHOT_PATH / 'prediction_shadow.txt').read()
    assert result.stdout == expected_result


@pytest.mark.parametrize('measurements_per_observable, observable_path, expected_result', [
    (
        10,
        SNAPSHOT_PATH / 'generated_observables_5.txt',
        SNAPSHOT_PATH / 'data_acquisition_shadow_d_m10_system_size5.txt',
    ),
    (
        20,
        SNAPSHOT_PATH / 'generated_observables_10.txt',
        SNAPSHOT_PATH / 'data_acquisition_shadow_d_m20_system_size10.txt',
    ),
])
def test_data_acquisition_shadow(
    measurements_per_observable,
    observable_path,
    expected_result,
):
    result = runner.invoke(data_acquisition_shadow.app, ['derandomized', str(measurements_per_observable), str(observable_path)])

    expected_result = open(expected_result).read()
    assert result.stdout == expected_result


@pytest.mark.parametrize('system_size, expected_result', [
    (5, SNAPSHOT_PATH / 'generated_observables_5.txt'),
    (10, SNAPSHOT_PATH / 'generated_observables_10.txt'),
    (20, SNAPSHOT_PATH / 'generated_observables_20.txt'),
])
def test_generate_observables(system_size, expected_result):
    result = runner.invoke(generate_observables.app, f'{system_size}')

    assert result.exit_code == 0

    expected_result = open(expected_result).read()
    assert result.stdout == expected_result
