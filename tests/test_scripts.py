import contextlib
import io
import pathlib
import runpy
import sys
from unittest.mock import patch

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
])
def test_data_acquisition_shadow(
    measurements_per_observable,
    observable_path,
    expected_result,
):
    with (
        patch.object(sys, 'argv', [data_acquisition_shadow.__name__, '-d', str(measurements_per_observable), str(observable_path)]),
        contextlib.redirect_stdout(io.StringIO()) as stream,
    ):
        runpy.run_module(data_acquisition_shadow.__name__, run_name='__main__')

    expected_result = open(expected_result).read()
    assert stream.getvalue() == expected_result


@pytest.mark.parametrize('system_size, expected_result', [
    (5, SNAPSHOT_PATH / 'generated_observables_5.txt'),
    (20, SNAPSHOT_PATH / 'generated_observables_20.txt'),
])
def test_generate_observables(system_size, expected_result):
    result = runner.invoke(generate_observables.app, f'{system_size}')

    assert result.exit_code == 0

    expected_result = open(expected_result).read()
    assert result.stdout == expected_result
