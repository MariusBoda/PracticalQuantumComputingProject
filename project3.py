import stim
import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bp_decoder import BpDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.union_find_decoder import UnionFindDecoder
from ldpc.belief_find_decoder import BeliefFindDecoder
from pymatching import Matching
import matplotlib.pyplot as plt
from beliefmatching import detector_error_model_to_check_matrices
import csv
import os

def generate_example_tasks(modified=False):
    for p in np.arange(0.001, 0.1, 0.01):
        for d in [3, 5]:
            if modified:
                before_measure_flip_probability = p * 5
            else:
                before_measure_flip_probability = p

            sc_circuit = stim.Circuit.generated(
                rounds=d,
                distance=d,
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=before_measure_flip_probability,
                before_round_data_depolarization=p,
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

def decode_and_plot(decoder_type, subplot_pos, load_from_file=False):
    results_unmodified = []
    results_modified = []
    num_shots = 10000

    unmodified_filename = f'results_unmodified_{decoder_type}.csv'
    modified_filename = f'results_modified_{decoder_type}.csv'

    if load_from_file and os.path.exists(unmodified_filename) and os.path.exists(modified_filename):
        results_unmodified = load_results_from_csv(unmodified_filename)
        results_modified = load_results_from_csv(modified_filename)
    else:
        for modified in [False, True]:
            for index, task in enumerate(generate_example_tasks(modified=modified)):
                print(f"Task # {index + 1}: p = {task['p']}, d = {task['d']}, modified = {modified}")
                circuit = task['circuit']

                sampler = circuit.compile_detector_sampler()
                detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
                dem = circuit.detector_error_model(decompose_errors=True)
                matrices = detector_error_model_to_check_matrices(dem)

                if decoder_type == 'Bp':
                    decoder = BpDecoder(
                        matrices.check_matrix,
                        error_channel=list(matrices.priors),
                        max_iter=20,
                        bp_method="ms",
                        ms_scaling_factor=0.625,
                        schedule="parallel"
                    )
                    
                elif decoder_type == 'BpOsd':
                    decoder = BpOsdDecoder(
                        matrices.check_matrix,
                        error_channel=list(matrices.priors),
                        max_iter=10,
                        bp_method="ms",
                        ms_scaling_factor=0.625,
                        schedule="parallel",
                        omp_thread_count=1,
                        osd_method="osd0",
                        osd_order=0
                    )
                elif decoder_type == 'MWPM':
                    decoder = Matching.from_detector_error_model(
                        dem
                    )
                elif decoder_type == 'BpLsd':
                    decoder = BpLsdDecoder(
                        matrices.check_matrix,
                        error_channel=list(matrices.priors),
                        max_iter=20,
                        bp_method="ms",
                        ms_scaling_factor=0.625,
                        schedule="parallel",
                        lsd_order=0
                    )
                elif decoder_type == 'BpUnionFind':
                    decoder = BeliefFindDecoder(
                        matrices.check_matrix,
                        error_channel=list(matrices.priors),
                        max_iter=10,
                        bp_method="ms",
                        ms_scaling_factor=0.625,
                        schedule="parallel",
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
                if modified:
                    results_modified.append(result)
                else:
                    results_unmodified.append(result)

        save_results_to_csv(results_unmodified, unmodified_filename)
        save_results_to_csv(results_modified, modified_filename)

    ax = plt.subplot(1, 3, subplot_pos)
    for d in [3, 5]:
        data_unmodified = [result for result in results_unmodified if result['d'] == str(d)]
        ps_unmodified = [float(result['p']) for result in data_unmodified]
        error_rates_unmodified = [float(result['logical_error_rate']) for result in data_unmodified]

        data_modified = [result for result in results_modified if result['d'] == str(d)]
        ps_modified = [float(result['p']) for result in data_modified]
        error_rates_modified = [float(result['logical_error_rate']) for result in data_modified]

        ax.plot(ps_unmodified, error_rates_unmodified, 'o-', label=f'd={d}, unmodified', color='grey', alpha=0.5)
        ax.plot(ps_modified, error_rates_modified, 'o-', label=f'd={d}, modified')

    ax.set_ylabel('Logical Error Rate')
    ax.set_xlabel('Physical Error Rate')
    ax.set_title(decoder_type)
    ax.legend()
    ax.loglog()
    ax.grid(True, which="both", ls="--", linewidth=0.5)

def main(load_from_file=False):
    fig = plt.figure(figsize=(18, 6))
    decode_and_plot('MWPM', 1, load_from_file=load_from_file)
    decode_and_plot('BpOsd', 2, load_from_file=load_from_file)  # Plot in the middle
    decode_and_plot('BpUnionFind', 3, load_from_file=load_from_file)  # Plot on the right
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main(load_from_file=True)
