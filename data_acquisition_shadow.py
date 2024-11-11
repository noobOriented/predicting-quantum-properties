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
import functools
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
        all_observables: list[dict[int, PauliOp]] = [
            {
                int(position): t.cast(PauliOp, pauli_XYZ)
                for pauli_XYZ, position in more_itertools.chunked(line.split()[1:], 2)
            }
            for line in f
        ]

    with (cProfile.Profile() if profile else contextlib.nullcontext()) as p:
        measurement_procedure = derandomized_classical_shadow(all_observables, num_of_measurements_per_observable, system_size)
        for measurement in measurement_procedure:
            print(' '.join(measurement))
    
    if p:
        pstats.Stats(p).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)


def derandomized_classical_shadow(
    observables: t.Sequence[Observable],
    num_of_measurements_per_observable: int,
    system_size: int,
    weight: t.Sequence[float] | None = None,
) -> t.Iterator[list[PauliOp]]:
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

    def cost_function(
        num_of_matches_in_this_round: t.Mapping[int, int],
        /, *,
        eta: float = 0.9,  # a hyperparameter subject to change
    ):
        nu = 1 - math.exp(-eta / 2)

        for i in needed_to_measure:
            matches_needed = len(observables[i]) - num_of_matches_in_this_round[i]
            v = (
                -num_of_measurements_so_far[i] * eta / 2
                + (math.log(1 - nu / (3 ** matches_needed)) if matches_needed <= system_size else 0)
            )
            yield v / weight[i]

    @functools.cache
    def get_column(pos: int) -> dict[int, PauliOp]:
        return {
            i: pauli
            for i, ob in enumerate(observables)
            if (pauli := ob.get(pos))
        }

    @functools.cache
    def dense_matchup(pos: int, pauli: PauliOp, /):
        # find observables which have op on pos
        return {
            i: match(pauli, target_pauli)
            for i, target_pauli in get_column(pos).items()
        }

    def match(x: PauliOp, y: PauliOp):
        return 1 if x == y else -100 * (system_size + 10)  # impossible to measure

    shift = 0
    needed_to_measure = set(range(len(observables)))
    num_of_measurements_so_far = collections.defaultdict[int, int](int)

    for _ in range(num_of_measurements_per_observable * len(observables)):
        # A single round of parallel measurement over "system_size" number of qubits
        num_of_matches_in_this_round = collections.defaultdict[int, int](int)
        single_round_measurement: list[PauliOp] = []
        total_log_values: list[float] = []

        for pos in range(system_size):
            # for each qubit, picks the best pauli op & trace all log values to update shift
            best_cost = float('inf')

            for pauli in PAULI_OPS:
                # Assume the dice rollout to be "dice_roll_pauli"
                attempt = {
                    i: num_of_matches_in_this_round[i] + dense_matchup(pos, pauli).get(i, 0)
                    for i in needed_to_measure
                }
                logits = list(cost_function(attempt))  # cost of pos, pauli
                total_log_values += logits
                cost = compute_mean([math.exp(lv - shift) for lv in logits])
                if cost < best_cost:
                    best_cost = cost
                    best_sol = attempt
                    best_pauli = pauli

            single_round_measurement.append(best_pauli)
            num_of_matches_in_this_round = best_sol

        yield single_round_measurement

        # Update num_of_measurements_so_far
        for i, matches in num_of_matches_in_this_round.items():
            if matches == len(observables[i]):  # finished measuring all qubits
                num_of_measurements_so_far[i] += 1

        for i, measurements in num_of_measurements_so_far.items():
            if measurements >= math.floor(weight[i] * num_of_measurements_per_observable):
                needed_to_measure -= {i}

        if not needed_to_measure:
            return

        shift = compute_mean(total_log_values, 0)


def compute_mean(seq: t.Sequence[float], default=0) -> float:
    if not seq:
        return default

    return altsum(seq) / len(seq)  # FIXME not equivalent to sum(log_values) due to rounding


def altsum(seq: t.Iterable[float]) -> float:
    x = 0.0  # FIXME not equivalent to sum(log_values) due to rounding
    for v in seq:
        x += v
    return x


if __name__ == '__main__':
    app()
