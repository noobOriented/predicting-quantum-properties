import pytest
from typer.testing import CliRunner

import prediction_shadow


runner = CliRunner()


def test_shadow():
    result = runner.invoke(prediction_shadow.app, ['measurement.txt', 'observables.txt'])

    predictions = [float(line) for line in result.stdout.splitlines()]

    assert predictions == pytest.approx(EXPECTED_PREDICTIONS)


EXPECTED_PREDICTIONS = [
    0.04901511681172698,
    0.006747638326585695,
    0.014625228519195612,
    -0.005203816131830009,
    0.017512348450830714,
    -0.024036281179138322,
    -0.02204585537918871,
    -0.002697841726618705,
    -0.038271049076992344,
    0.003200731595793324,
    -0.023298309730470534,
    -0.0031545741324921135,
    -0.0031890660592255125,
    0.0022471910112359553,
    1.0,
    1.0,
]
