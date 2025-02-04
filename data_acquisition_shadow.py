#
# This code is created by Hsin-Yuan Huang (https://momohuang.github.io/).
# For more details, see the accompany paper:
#  "Predicting Many Properties of a Quantum System from Very Few Measurements".
# This Python version is slower than the C++ version. (there are less code optimization)
# But it should be easier to understand and build upon.
#

import contextlib
import cProfile
import pathlib
import pstats
import random
import typing as t

import more_itertools
import numpy as np
import numpy.typing as npt
import typer


app = typer.Typer()


PauliOp = t.Literal['X', 'Y', 'Z']
PAULI_OPS: list[PauliOp] = ['X', 'Y', 'Z']
PAULI_TO_INT: t.Mapping[PauliOp, int] = {'X': 1, 'Y': 2, 'Z': 3}


@app.command(
    'randomized',
    help='This is the randomized version of classical shadow.'
        'We would output a list of Pauli measurements for the given [system size]'
        'with a total of [number of total measurements] repetitions.',
)
def randomized_classical_shadow(
    num_total_measurements: t.Annotated[int, typer.Argument(help='for the total number of measurement rounds')],
    system_size: t.Annotated[int, typer.Argument(help='for how many qubits in the quantum system')],
):
    for _ in range(num_total_measurements):
        print(' '.join(random.choice(PAULI_OPS) for _ in range(system_size)))  # noqa: S311


@app.command(
    'derandomized',
    help='This is the derandomized version of classical shadow.'
        'We would output a list of Pauli measurements to measure all observables'
        'in [observable.txt] for at least [number of measurements per observable] times.',
)
def derandomized_classical_shadow_command(
    num_of_measurements_per_observable: int,
    observable_file: pathlib.Path,
    profile: bool = False,
):
    with (cProfile.Profile() if profile else contextlib.nullcontext()) as p:
        observables = _parse_observables(observable_file)
        measurement_procedure = derandomized_classical_shadow(observables, num_of_measurements_per_observable)
        for measurement in measurement_procedure:
            print(' '.join(PAULI_OPS[idx] for idx in measurement))

    if p:
        pstats.Stats(p).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)


def _parse_observables(path):
    with open(path) as f:
        system_size = int(f.readline())
        lines = f.readlines()
    
    out = np.zeros((len(lines), system_size), dtype=np.uint8)
    for line, row in zip(lines, out):
        line = line.split()
        indices = [int(s) for s in line[2::2]]
        values = [PAULI_TO_INT[s] for s in line[1::2]]
        row[indices] = values

    return out


def derandomized_classical_shadow(
    observables: npt.NDArray[np.uint],  # (N, Q)
    num_of_measurements_per_observable: int,
):
    observable_counts = np.count_nonzero(observables, axis=1)
    num_of_measurements = np.zeros([len(observables)])

    for _ in range(num_of_measurements_per_observable * len(observables)):
        measurement, finished_qubits = fit_measurement(
            num_of_measurements,
            observables,
            observable_counts,
        )
        if len(finished_qubits) == 0:
            raise RuntimeError('endless loop')

        yield measurement

        num_of_measurements[finished_qubits] += 1
        keep_indices, = np.nonzero(num_of_measurements < num_of_measurements_per_observable)
        if len(keep_indices) == 0:
            return

        num_of_measurements = num_of_measurements[keep_indices]
        observables = observables[keep_indices]
        observable_counts = observable_counts[keep_indices]


def fit_measurement(
    n_measurements: npt.NDArray,  # (N,)
    observables: npt.NDArray,  # shape (N, Q), value in [0, 4)
    observable_counts: npt.NDArray | None = None,
):
    if observable_counts is None:
        observable_counts = np.count_nonzero(observables, axis=1)

    matches = np.zeros([len(observables)])  # shape (N, )
    measurement: list[int] = []

    # find best op for each qubit
    for pos in range(observables.shape[1]):
        # 1. When observables[i, pos] == 0, its contribution to cost is independent to op thus can be ignored
        # 2. Once matches[i] becomes -inf, logits never change no matter what op you choose
        indices, = np.nonzero((observables[:, pos] != 0) & (matches != -np.inf))  # shape (M, )
        diff_matches = np.where(
            observables[indices, pos] == np.arange(1, 4)[:, np.newaxis],
            1,
            -np.inf,
        )  # shape (3, M)
        new_matches = matches[indices] + diff_matches  # shape (3, M)
        cost = cost_func(
            n_measurements[indices],  # shape (M,)
            observable_counts[indices] - new_matches,  # shape (3, M)
        )

        op_idx = np.argmin(cost)  # scalar in [0, 3)
        matches[indices] = new_matches[op_idx]
        measurement.append(int(op_idx))

    finished_qubits, = np.nonzero(matches == observable_counts)
    return measurement, finished_qubits



def cost_func(
    n_measurements: npt.NDArray,  # shape (N,)
    matches_needed: npt.NDArray,  # shape (3, N)
    *,
    eta: float = 0.9,
) -> npt.NDArray:  # shape (3,)
    nu = 1 - np.exp(-eta / 2)
    return (1 - nu / (3 ** matches_needed)) @ np.exp(-n_measurements * (eta / 2))


if __name__ == '__main__':
    app()
