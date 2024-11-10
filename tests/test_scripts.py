import pathlib

import pytest
from typer.testing import CliRunner

import generate_observables
import prediction_shadow


runner = CliRunner()

SNAPSHOT_PATH = pathlib.Path(__file__).parent / 'snapshots'


def test_prediction_shadow():
    result = runner.invoke(prediction_shadow.app, ['measurement.txt', 'observables.txt'])

    assert result.exit_code == 0

    expected_result = open(SNAPSHOT_PATH / 'prediction_shadow.txt').read()
    assert result.stdout == expected_result


@pytest.mark.parametrize('system_size, expected_result', [
    (5, SNAPSHOT_PATH / 'generated_observables_5.txt'),
    (20, SNAPSHOT_PATH / 'generated_observables_20.txt'),
])
def test_generate_observables(system_size, expected_result):
    result = runner.invoke(generate_observables.app, f'{system_size}')

    assert result.exit_code == 0

    expected_result = open(expected_result).read()
    assert result.stdout == expected_result
