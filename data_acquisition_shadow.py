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

import more_itertools
import typer


app = typer.Typer()


@app.command(
    '-r',
    help='This is the randomized version of classical shadow.'
        'We would output a list of Pauli measurements for the given [system size]'
        'with a total of [number of total measurements] repetitions.',
)
def randomized_classical_shadow(num_total_measurements: int, system_size: int):
    #
    # Implementation of the randomized classical shadow
    #
    #    num_total_measurements: int for the total number of measurement rounds
    #    system_size: int for how many qubits in the quantum system
    #
    for t in range(num_total_measurements):
        single_round_measurement = [random.choice(['X', 'Y', 'Z']) for i in range(system_size)]
        print(' '.join(single_round_measurement))


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
        all_observables = [
            [
                (pauli_XYZ, int(position))
                for pauli_XYZ, position in more_itertools.chunked(line.split()[1:], 2)
            ]
            for line in f
        ]

    measurement_procedure = derandomized_classical_shadow(all_observables, num_of_measurements_per_observable, system_size)
    for measurement in measurement_procedure:
        print(' '.join(measurement))


def derandomized_classical_shadow(
    all_observables: list[list[tuple[str, int]]],
    num_of_measurements_per_observable: int,
    system_size: int,
    weight=None,
) -> list[list[str]]:
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
    assert(len(weight) == len(all_observables))

    sum_log_value = 0
    sum_cnt = 0

    def cost_function(num_of_measurements_so_far, num_of_matches_needed_in_this_round, shift = 0):
        eta = 0.9 # a hyperparameter subject to change
        nu = 1 - math.exp(-eta / 2)

        nonlocal sum_log_value
        nonlocal sum_cnt

        cost = 0
        for i, zipitem in enumerate(zip(num_of_measurements_so_far, num_of_matches_needed_in_this_round)):
            measurement_so_far, matches_needed = zipitem
            if num_of_measurements_so_far[i] >= math.floor(weight[i] * num_of_measurements_per_observable):
                continue

            if system_size < matches_needed:
                V = eta / 2 * measurement_so_far
            else:
                V = eta / 2 * measurement_so_far - math.log(1 - nu / (3 ** matches_needed))
            cost += math.exp(-V / weight[i] - shift)

            sum_log_value += V / weight[i]
            sum_cnt += 1

        return cost

    def match_up(qubit_i, dice_roll_pauli, single_observable):
        for pauli, pos in single_observable:
            if pos != qubit_i:
                continue
            else:
                if pauli != dice_roll_pauli:
                    return -1
                else:
                    return 1
        return 0

    num_of_measurements_so_far = [0] * len(all_observables)
    measurement_procedure = []

    for repetition in range(num_of_measurements_per_observable * len(all_observables)):
        # A single round of parallel measurement over "system_size" number of qubits
        num_of_matches_needed_in_this_round = [len(P) for P in all_observables]
        single_round_measurement = []

        shift = sum_log_value / sum_cnt if sum_cnt > 0 else 0;
        sum_log_value = 0.0
        sum_cnt = 0

        for qubit_i in range(system_size):
            cost_of_outcomes = dict([('X', 0), ('Y', 0), ('Z', 0)])

            for dice_roll_pauli in ['X', 'Y', 'Z']:
                # Assume the dice rollout to be "dice_roll_pauli"
                for i, single_observable in enumerate(all_observables):
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1:
                        num_of_matches_needed_in_this_round[i] += 100 * (system_size+10) # impossible to measure
                    if result == 1:
                        num_of_matches_needed_in_this_round[i] -= 1 # match up one Pauli X/Y/Z

                cost_of_outcomes[dice_roll_pauli] = cost_function(num_of_measurements_so_far, num_of_matches_needed_in_this_round, shift=shift)

                # Revert the dice roll
                for i, single_observable in enumerate(all_observables):
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1:
                        num_of_matches_needed_in_this_round[i] -= 100 * (system_size+10) # impossible to measure
                    if result == 1:
                        num_of_matches_needed_in_this_round[i] += 1 # match up one Pauli X/Y/Z

            for dice_roll_pauli in ['X', 'Y', 'Z']:
                if min(cost_of_outcomes.values()) < cost_of_outcomes[dice_roll_pauli]:
                    continue
                # The best dice roll outcome will come to this line
                single_round_measurement.append(dice_roll_pauli)
                for i, single_observable in enumerate(all_observables):
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1:
                        num_of_matches_needed_in_this_round[i] += 100 * (system_size+10) # impossible to measure
                    if result == 1:
                        num_of_matches_needed_in_this_round[i] -= 1 # match up one Pauli X/Y/Z
                break

        measurement_procedure.append(single_round_measurement)

        for i, single_observable in enumerate(all_observables):
            if num_of_matches_needed_in_this_round[i] == 0: # finished measuring all qubits
                num_of_measurements_so_far[i] += 1

        success = 0
        for i, single_observable in enumerate(all_observables):
            if num_of_measurements_so_far[i] >= math.floor(weight[i] * num_of_measurements_per_observable):
                success += 1

        if success == len(all_observables):
            break

    return measurement_procedure


if __name__ == '__main__':
    app()
