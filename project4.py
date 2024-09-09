import stim
import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bp_decoder import BpDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.belief_find_decoder import BeliefFindDecoder
from pymatching import Matching
import matplotlib.pyplot as plt
from beliefmatching import detector_error_model_to_check_matrices
import csv
import os

def generate_example_tasks():
    for p in np.arange(0.001, 0.1, 0.01):
        for d in [3, 5]:
            before_measure_flip_probability = p
            after_reset_flip_probability = p
            before_round_data_depolarization = p
            after_clifford_depolarization = p

            sc_circuit = stim.Circuit.generated(
                rounds=d,
                distance=d,
                after_clifford_depolarization=after_clifford_depolarization,
                after_reset_flip_probability=after_reset_flip_probability,
                before_measure_flip_probability=before_measure_flip_probability,
                before_round_data_depolarization=before_round_data_depolarization,
                code_task=f'surface_code:rotated_memory_z',
            )
            yield {'circuit': sc_circuit, 'p': p, 'd': d, 'rounds': d}

def save_results_to_csv(results, filename):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

def load_results_from_csv(filename):
    with open(filename, 'r') as input_file:
        reader = csv.DictReader(input_file)
        return list(reader)

def decode_and_plot(decoder_type, subplot_pos, schedule_types, load_from_file=False):
    results = {schedule: [] for schedule in schedule_types}
    ms_scaling_factor = 0.625  # Default scaling factor
    num_shots = 5000

    filenames = {schedule: f'results_{decoder_type}_{schedule}.csv' for schedule in schedule_types}

    if load_from_file and all(os.path.exists(filename) for filename in filenames.values()):
        for schedule in schedule_types:
            results[schedule] = load_results_from_csv(filenames[schedule])
    else:
        for schedule_type in schedule_types:
            for index, task in enumerate(generate_example_tasks()):
                print(f"Task # {index + 1}: p = {task['p']}, d = {task['d']}, schedule = {schedule_type}")
                circuit = task['circuit']

                sampler = circuit.compile_detector_sampler()
                detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
                dem = circuit.detector_error_model(decompose_errors=True)
                matrices = detector_error_model_to_check_matrices(dem)

                if decoder_type == 'BpOsd':
                    decoder = BpOsdDecoder(
                        matrices.check_matrix,
                        error_channel=list(matrices.priors),
                        max_iter=10,
                        bp_method="ms",
                        ms_scaling_factor=ms_scaling_factor,
                        schedule=schedule_type,
                        omp_thread_count=1,
                        osd_method="osd0",
                        osd_order=0
                    )
                elif decoder_type == 'BpLsd':
                    decoder = BpLsdDecoder(
                        matrices.check_matrix,
                        error_channel=list(matrices.priors),
                        max_iter=10,
                        bp_method="ms",
                        ms_scaling_factor=ms_scaling_factor,
                        schedule=schedule_type,
                        lsd_order=0
                    )
                elif decoder_type == 'BpUnionFind':
                    decoder = BeliefFindDecoder(
                        matrices.check_matrix,
                        error_channel=list(matrices.priors),
                        max_iter=10,
                        bp_method="ms",
                        ms_scaling_factor=ms_scaling_factor,
                        schedule=schedule_type,
                        uf_method="inversion"
                    )
                else:
                    raise ValueError("Unknown decoder type")

                predictions = np.zeros((num_shots, matrices.check_matrix.shape[1]), dtype=bool)

                for i in range(num_shots):
                    predictions[i, :] = decoder.decode(detection_events[i, :])

                logical_num_errors = ((predictions @ matrices.observables_matrix.toarray().T) + observable_flips) % 2
                logical_num_errors = np.any(logical_num_errors, axis=1)

                logical_num_errors = logical_num_errors.sum()
                logical_error_rate = logical_num_errors / num_shots
                ler_round = 1 - (1 - logical_error_rate) ** (1/task['rounds'])

                result = {'p': task['p'], 'd': task['d'], 'rounds': task['rounds'], 'logical_error_rate': ler_round}
                results[schedule_type].append(result)

            save_results_to_csv(results[schedule_type], filenames[schedule_type])

    ax = plt.subplot(1, 3, subplot_pos)
    for schedule_type in schedule_types:
        for d in [3, 5]:
            data = [result for result in results[schedule_type] if result['d'] == str(d)]
            ps = [float(result['p']) for result in data]
            error_rates = [float(result['logical_error_rate']) for result in data]

            color = 'grey' if schedule_type == 'parallel' else None
            label = f'd={d}, schedule={schedule_type}'
            ax.plot(ps, error_rates, 'o-', label=label, color=color, alpha=0.5 if schedule_type == 'parallel' else 1)

    ax.set_ylabel('Logical Error Rate')
    ax.set_xlabel('Physical Error Rate')
    ax.set_title(f"{decoder_type}")
    ax.legend()
    ax.loglog()
    ax.grid(True, which="both", ls="--", linewidth=0.5)

def main(load_from_file=False):
    fig = plt.figure(figsize=(18, 6))
    schedule_types = ['parallel', 'serial']  # Schedule types to iterate over

    decode_and_plot('BpOsd', 1, schedule_types, load_from_file=load_from_file)
    #decode_and_plot('BpLsd', 2, schedule_types, load_from_file=load_from_file)  # Plot in the middle
    #decode_and_plot('BpUnionFind', 3, schedule_types, load_from_file=load_from_file)  # Plot on the right

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main(load_from_file=True)
