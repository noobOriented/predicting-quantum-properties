#
# This code is created by Hsin-Yuan Huang (https://momohuang.github.io/).
# For more details, see the accompany paper:
#  "Predicting Many Properties of a Quantum System from Very Few Measurements".
# This Python version is slower than the C++ version. (there are less code optimization)
# But it should be easier to understand and build upon.
#

import collections
import contextlib
import cProfile
import math
import pathlib
import pstats
import random
import typing as t

import more_itertools
import numpy as np
import typer


app = typer.Typer()


PauliOp = t.Literal['X', 'Y', 'Z']
PAULI_OPS: list[PauliOp] = ['X', 'Y', 'Z']
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
    weight: t.Sequence[float] | None = None,
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

    def cost_function(new_matches: t.Mapping[int, float]):
        eta = 0.9  # a hyperparameter subject to change
        nu = 1 - math.exp(-eta / 2)

        for i in needed_to_measure:
            matches_needed = len(observables[i]) - new_matches[i]
            v = (
                -num_of_measurements_so_far[i] * eta / 2  # const over qubit
                + (math.log(1 - nu / (3 ** matches_needed)) if not math.isinf(matches_needed) else 0)
            )
            yield v / weight[i] - shift

    shift = 0
    needed_to_measure = set(range(len(observables)))
    num_of_measurements_so_far = collections.defaultdict[int, int](int)

    for _ in range(num_of_measurements_per_observable * len(observables)):
        # A single round of parallel measurement over "system_size" number of qubits
        num_of_matches_in_this_round = collections.defaultdict[int, int](int)
        single_round_measurement: list[PauliOp] = []

        for pos in range(system_size):
            best_cost = float('inf')

            # for each qubit, picks the best measurement
            for pauli in PAULI_OPS:
                # Assume the dice rollout to be "dice_roll_pauli"
                new_matches = {
                    i: num_of_matches_in_this_round[i] + compute_match_score(pauli, observables[i].get(pos))
                    for i in needed_to_measure
                }
                # logit decreases when num_of_matches_in_this_round increases
                # add shift term to prevent float underflow
                cost = np.mean(np.exp(np.asarray(list(cost_function(new_matches)))))
                if cost < best_cost:
                    best_cost = cost
                    best_sol = new_matches
                    best_pauli = pauli

            shift = np.log(best_cost)
            single_round_measurement.append(best_pauli)
            num_of_matches_in_this_round = best_sol

        yield single_round_measurement

        # Update num_of_measurements_so_far
        finished_qubits = [
            i
            for i, matches in num_of_matches_in_this_round.items()
            if matches == len(observables[i])  # finished measuring all qubits
        ]
        if not finished_qubits:
            raise RuntimeError('endless loop')
        
        for i in finished_qubits:
            num_of_measurements_so_far[i] += 1

        needed_to_measure -= {
            i
            for i, measurements in num_of_measurements_so_far.items()
            if measurements >= weight[i] * num_of_measurements_per_observable
        }

        if not needed_to_measure:
            return


def compute_match_score(a: PauliOp, b: PauliOp | None):
    if b is None:
        return 0
    if a == b:
        return 1
    return float('-inf')  # FIXME


if __name__ == '__main__':
    app()
