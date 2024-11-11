#
# This code is created by Hsin-Yuan Huang (https://momohuang.github.io/).
# For more details, see the accompany paper:
#  "Predicting Many Properties of a Quantum System from Very Few Measurements".
# This Python version is slower than the C++ version. (there are less code optimization)
# But it should be easier to understand and build upon.
#
import math
import pathlib
import random
import typing as t

import more_itertools
import typer


app = typer.Typer()


PauliOp = t.Literal['X', 'Y', 'Z']
PAULI_OPS: list[PauliOp] = ['X', 'Y', 'Z']


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
):
    with open(observable_file) as f:
        system_size = int(f.readline())
        all_observables: list[dict[int, PauliOp]] = [
            {
                int(position): pauli_XYZ
                for pauli_XYZ, position in more_itertools.chunked(line.split()[1:], 2)
            }
            for line in f
        ]

    measurement_procedure = derandomized_classical_shadow(all_observables, num_of_measurements_per_observable, system_size)
    for measurement in measurement_procedure:
        print(' '.join(measurement))


def derandomized_classical_shadow(
    all_observables: t.Sequence[t.Mapping[int, PauliOp]],
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
        weight = [1.0] * len(all_observables)
    elif len(weight) != len(all_observables):
        raise ValueError

    def cost_function(
        num_of_measurements_so_far: t.Sequence[int],
        num_of_matches_in_this_round: t.Sequence[int],
        /, *,
        eta: float = 0.9,  # a hyperparameter subject to change
    ) -> t.Iterator[float]:
        nu = 1 - math.exp(-eta / 2)

        for measurement_so_far, matches, w, ob in zip(
            num_of_measurements_so_far,
            num_of_matches_in_this_round,
            weight,
            all_observables,
        ):
            if measurement_so_far >= math.floor(w * num_of_measurements_per_observable):
                continue

            v = measurement_so_far * eta / 2
            matches_needed = len(ob) - matches
            if system_size >= matches_needed:
                v -= math.log(1 - nu / (3 ** matches_needed))
            yield v / w

    def match_up(pos: int, dice_roll_pauli: PauliOp, ob: t.Mapping[int, PauliOp], /) -> int:
        target_pauli = ob.get(pos)
        if not target_pauli:
            return 0
        if target_pauli == dice_roll_pauli:
            return 1
        return -100 * (system_size + 10)  # impossible to measure

    shift = 0
    num_of_measurements_so_far = [0] * len(all_observables)

    for _ in range(num_of_measurements_per_observable * len(all_observables)):
        # A single round of parallel measurement over "system_size" number of qubits
        num_of_matches_in_this_round = [0] * len(all_observables)
        single_round_measurement: list[PauliOp] = []
        total_log_values: list[float] = []

        for pos in range(system_size):
            # for each qubit, picks the best pauli op & trace all log values to update shift
            best_cost = float('inf')

            for pauli in PAULI_OPS:
                # Assume the dice rollout to be "dice_roll_pauli"
                attempt = [
                    x + match_up(pos, pauli, ob)
                    for x, ob in zip(num_of_matches_in_this_round, all_observables)
                ]
                log_values = list(cost_function(num_of_measurements_so_far, attempt))
                cost = altsum(math.exp(-lv - shift) for lv in log_values)
                if cost < best_cost:
                    best_cost = cost
                    best_sol = attempt
                    best_pauli = pauli

                total_log_values += log_values

            single_round_measurement.append(best_pauli)
            num_of_matches_in_this_round[:] = best_sol

        yield single_round_measurement

        for i, (match, ob) in enumerate(zip(num_of_matches_in_this_round, all_observables)):
            if match == len(ob): # finished measuring all qubits
                num_of_measurements_so_far[i] += 1

        success = sum(
            1
            for i, _ in enumerate(all_observables)
            if num_of_measurements_so_far[i] >= math.floor(weight[i] * num_of_measurements_per_observable)
        )
        if success == len(all_observables):
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
