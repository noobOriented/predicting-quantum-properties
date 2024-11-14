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
Observable = t.Mapping[int, PauliOp]


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
    with open(observable_file) as f:
        system_size = int(f.readline())
        observables: list[Observable] = [
            {
                int(position): t.cast(PauliOp, pauli_XYZ)
                for pauli_XYZ, position in more_itertools.chunked(line.split()[1:], 2)
            }
            for line in f
        ]

    with (cProfile.Profile() if profile else contextlib.nullcontext()) as p:
        measurement_procedure = derandomized_classical_shadow(observables, num_of_measurements_per_observable, system_size)
        for measurement in measurement_procedure:
            print(' '.join(measurement))

    if p:
        pstats.Stats(p).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)


def derandomized_classical_shadow(
    observables: t.Sequence[Observable],
    num_of_measurements_per_observable: int,
    system_size: int,
    weight: t.Sequence[float] | npt.NDArray | None = None,
    eta: float = 0.9  # a hyperparameter subject to change,
):
    #
    # Implementation of the derandomized classical shadow
    #
    #     all_observables: a list of Pauli observables, each Pauli observable is a list of tuple
    #                      of the form ("X", position) or ("Y", position) or ("Z", position)
    #     num_of_measurements_per_observable: int for the number of measurement for each observable
    #     system_size: int for how many qubits in the quantum system
    #     weight: None or a list of coefficients for each observable
    #             None -- neglect this parameter
    #             a list -- modify the number of measurements for each observable by the corresponding weight
    #
    if weight is None:
        weight = [1.0] * len(observables)
    elif len(weight) != len(observables):
        raise ValueError

    dense_observables = np.zeros([len(observables), system_size], dtype=np.uint8)
    for i, ob in enumerate(observables):
        for pos, pauli in ob.items():
            dense_observables[i, pos] = PAULI_TO_INT[pauli]

    weight = np.asarray(weight)
    len_obs = np.asarray([len(ob) for ob in observables])
    num_of_measurements_so_far = np.zeros([len(observables)])

    for _ in range(num_of_measurements_per_observable * len(observables)):
        # A single round of parallel measurement over "system_size" number of qubits
        matches_in_this_round = np.zeros([len(dense_observables)])
        single_round_measurement: list[PauliOp] = []

        for pos in range(system_size):
            best_cost = np.inf

            # for each qubit, picks the best measurement
            for op in PAULI_OPS:
                # for observable have no `op` on `pos`, it contributes nothing to the cost, just ignore it
                indices = np.nonzero(dense_observables[:, pos] != 0)[0]
                new_matches = matches_in_this_round[indices] + np.where(
                    dense_observables[indices, pos] == PAULI_TO_INT[op],
                    1,
                    -np.inf,
                )
                nu = 1 - np.exp(-eta / 2)
                matches_needed = len_obs[indices] - new_matches
                logits = (
                     -num_of_measurements_so_far[indices] * eta / 2
                    + np.log(1 - nu / np.pow(3, matches_needed))  # decreases when more matches
                ) / weight[indices]
                cost = logsumexp(logits)
                if cost < best_cost:  # TODO refactor to min(iterable)
                    best_cost = cost
                    best_sol = (op, indices, new_matches)

            op, indices, new_matches = best_sol
            single_round_measurement.append(op)
            matches_in_this_round[indices] = new_matches

        yield single_round_measurement

        # Update num_of_measurements_so_far
        finished_qubits = np.nonzero(matches_in_this_round == len_obs)[0]
        if len(finished_qubits) == 0:
            raise RuntimeError('endless loop')

        num_of_measurements_so_far[finished_qubits] += 1
        keep_indices = np.nonzero(num_of_measurements_so_far < weight * num_of_measurements_per_observable)[0]
        if len(keep_indices) == 0:
            return

        num_of_measurements_so_far = num_of_measurements_so_far[keep_indices]
        dense_observables = dense_observables[keep_indices]
        len_obs = len_obs[keep_indices]
        weight = weight[keep_indices]


def logsumexp(logits: npt.NDArray):
    if len(logits) == 0:
        return 0
    # To prevent overflow or underflow
    shift = np.min(logits)
    return np.log(np.sum(np.exp(logits - shift))) + shift


def take_unordered(arr, indices):
    ...


if __name__ == '__main__':
    app()
