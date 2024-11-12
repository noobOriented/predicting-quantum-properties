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
        return [
            (
                -num_of_measurements_so_far[i] * eta / 2  # const over qubit
                + math.log(1 - nu / (3 ** (len(observables[i]) - m)))   # decrease when more matches -> less cost
            ) / weight[i]
            for i, m in new_matches.items()
        ]

    needed_to_measure = set(range(len(observables)))
    num_of_measurements_so_far = collections.defaultdict[int, int](int)

    @functools.cache
    def observable_have_op_at_pos(pos: int):
        return {i for i, ob in enumerate(observables) if ob.get(pos)}

    for _ in range(num_of_measurements_per_observable * len(observables)):
        # A single round of parallel measurement over "system_size" number of qubits
        num_of_matches_in_this_round = collections.defaultdict[int, int](int)
        single_round_measurement: list[PauliOp] = []

        for pos in range(system_size):
            best_cost = float('inf')

            # for each qubit, picks the best measurement
            for op in PAULI_OPS:
                new_matches = {
                    i: num_of_matches_in_this_round[i] + (1 if op == observables[i][pos] else -math.inf)  # XXX can't be match
                    # When there's no match on (i, pos), i-th term is constant over different choice of `op`
                    # thus ignore this i
                    for i in needed_to_measure & observable_have_op_at_pos(pos)
                }
                cost = logsumexp(cost_function(new_matches))
                if cost < best_cost:  # TODO refactor to min(iterable)
                    best_cost = cost
                    best_sol = new_matches
                    best_op = op

            single_round_measurement.append(best_op)
            num_of_matches_in_this_round |= best_sol

        yield single_round_measurement

        # Update num_of_measurements_so_far
        finished_qubits = [
            i
            for i, matches in num_of_matches_in_this_round.items()  # XXX some value will be -inf
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


def logsumexp(logits):
    if len(logits) == 0:
        return 0
    logits = np.asarray(logits)
    # To prevent overflow or underflow
    shift = np.min(logits)
    return np.log(np.sum(np.exp(logits - shift))) + shift


if __name__ == '__main__':
    app()
