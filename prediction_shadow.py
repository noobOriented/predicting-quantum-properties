import pathlib

import typer


app = typer.Typer()


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
        measurements = f.readlines()

    full_measurement = [
        [
            (pauli_XYZ, int(outcome))
            for pauli_XYZ, outcome in zip(line.split(' ')[0::2], line.split(' ')[1::2])
        ]
        for line in measurements
    ]

    with open(observable_path) as f:
        observable_size = int(f.readline())
        observables = f.readlines()

    for line in observables:
        one_observable = [
            (pauli_XYZ, int(position))
            for pauli_XYZ, position in zip(line.split(' ')[1::2], line.split(' ')[2::2])
        ]
        sum_product, cnt_match = estimate_exp(full_measurement, one_observable)
        print(sum_product / cnt_match)


def estimate_exp(
    full_measurement: list[list[tuple[str, int]]],
    one_observable: list[tuple[str, int]],
) -> tuple[int, int]:
    sum_product, cnt_match = 0, 0

    for single_measurement in full_measurement:
        not_match = 0
        product = 1

        for pauli_XYZ, position in one_observable:
            if pauli_XYZ != single_measurement[position][0]:
                not_match = 1
                break
            product *= single_measurement[position][1]
        if not_match == 1:
            continue

        sum_product += product
        cnt_match += 1

    return sum_product, cnt_match


if __name__ == '__main__':
    app()
