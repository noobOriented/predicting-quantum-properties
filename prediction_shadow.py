import math
import pathlib
import typing as t

import more_itertools
import typer


app = typer.Typer()
PauliOp = t.Literal['X', 'Y', 'Z']


@app.command(
    help='This option predicts the expectation of local observables.'
        'We would output the predicted value for each local observable given in [observable.txt]'
)
def main(
    measurement_path: pathlib.Path,
    observable_path: pathlib.Path,
):
    with open(measurement_path) as f:
        system_size = int(f.readline())
        full_measurement = [
            [
                (t.cast(PauliOp, pauli_XYZ), int(outcome))
                for pauli_XYZ, outcome in more_itertools.chunked(line.split(), 2)
            ]
            for line in f
        ]

    with open(observable_path) as f:
        observable_size = int(f.readline())
        observables = [
            [
                (t.cast(PauliOp, pauli_XYZ), int(position))
                for pauli_XYZ, position in more_itertools.chunked(line.split()[1:], 2)
            ]
            for line in f
        ]

    for one_observable in observables:
        sum_product, cnt_match = estimate_exp(full_measurement, one_observable)
        print(sum_product / cnt_match)


def estimate_exp(
    full_measurement: list[list[tuple[PauliOp, int]]],
    one_observable: list[tuple[PauliOp, int]],
) -> tuple[int, int]:
    sum_product, cnt_match = 0, 0

    for single_measurement in full_measurement:
        if not all(
            pauli_XYZ == single_measurement[position][0]
            for pauli_XYZ, position in one_observable
        ):
            continue

        product = math.prod(
            single_measurement[position][1]
            for _, position in one_observable
        )
        sum_product += product
        cnt_match += 1

    return sum_product, cnt_match


if __name__ == '__main__':
    app()
