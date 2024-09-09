import stim
import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder
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

def decode_and_plot(decoder_type, osd_method, subplot_pos, param_to_modify, param_values, load_from_file=False):
    results = []
    num_shots = 10000

    filename = f'results_{decoder_type}_{osd_method}_{param_to_modify}.csv'

    if load_from_file and os.path.exists(filename):
        results = load_results_from_csv(filename)
    else:
        for index, task in enumerate(generate_example_tasks()):
            print(f"Task # {index + 1}: p = {task['p']}, d = {task['d']}, osd_method = {osd_method}, param_to_modify = {param_to_modify}")
            circuit = task['circuit']

            sampler = circuit.compile_detector_sampler()
            detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
            dem = circuit.detector_error_model(decompose_errors=True)
            matrices = detector_error_model_to_check_matrices(dem)

            for param_value in param_values:
                decoder = BpOsdDecoder(
                    matrices.check_matrix,
                    error_channel=list(matrices.priors),
                    max_iter=10,
                    bp_method="ms",
                    ms_scaling_factor=0.625,
                    schedule="parallel",
                    omp_thread_count=1,
                    osd_method=osd_method,
                    osd_order=param_value
                )

                predictions = np.zeros((num_shots, matrices.check_matrix.shape[1]), dtype=bool)

                for i in range(num_shots):
                    predictions[i, :] = decoder.decode(detection_events[i, :])

                logical_num_errors = ((predictions @ matrices.observables_matrix.toarray().T) + observable_flips) % 2
                logical_num_errors = np.any(logical_num_errors, axis=1)

                logical_num_errors = logical_num_errors.sum()
                logical_error_rate = logical_num_errors / num_shots
                ler_round = 1 - (1 - logical_error_rate) ** (1/task['rounds'])

                result = {'p': task['p'], 'd': task['d'], 'rounds': task['rounds'], 'osd_method': osd_method, 'osd_order': param_value, 'logical_error_rate': ler_round}
                results.append(result)

        save_results_to_csv(results, filename)

    ax = plt.subplot(1, 3, subplot_pos)
    for d in [3, 5]:
        for param_value in param_values:
            data = [result for result in results if result['d'] == str(d) and result['osd_order'] == str(param_value)]
            ps = [float(result['p']) for result in data]
            error_rates = [float(result['logical_error_rate']) for result in data]

            ax.plot(ps, error_rates, 'o-', label=f'd={d}, osd_order={param_value}')

    ax.set_ylabel('Logical Error Rate')
    ax.set_xlabel('Physical Error Rate')
    ax.set_title(f"{decoder_type} ({osd_method})")
    ax.legend()
    ax.loglog()
    ax.grid(True, which="both", ls="--", linewidth=0.5)

def main(load_from_file=False):
    fig = plt.figure(figsize=(18, 6))
    decoder_type = 'BpOsd'
    osd_methods = ['osd_E']
    param_to_modify = 'osd_order'
    param_values_dict = {
        'osd_E': [0, 5, 15],
        #'osd_CS': [1]
    }

    for idx, osd_method in enumerate(osd_methods):
        decode_and_plot(decoder_type, osd_method, idx+1, param_to_modify, param_values_dict[osd_method], load_from_file=load_from_file)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main(load_from_file=True)
