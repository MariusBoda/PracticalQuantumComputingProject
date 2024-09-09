import stim
import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder
import matplotlib.pyplot as plt
from beliefmatching import detector_error_model_to_check_matrices

def generate_example_tasks():
    for p in np.arange(0.001, 0.01, 0.002):
        for d in [5, 7, 9]:
            sc_circuit = stim.Circuit.generated(
                rounds=d,
                distance=d,
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p,
                before_round_data_depolarization=p,
                code_task='surface_code:rotated_memory_z',
            )
            yield {'circuit': sc_circuit, 'p': p, 'd': d, 'rounds': d}


def decode(max_iter, 
           bp_method, 
           ms_scaling_factor, 
           schedule, 
           omp_thread_count, 
           osd_method, 
           osd_order, 
           dem, shots):
        
        matrices = detector_error_model_to_check_matrices(dem)
        bposd = BpOsdDecoder(
            matrices.check_matrix,
            error_channel=list(matrices.priors),
            max_iter=max_iter,
            bp_method=bp_method,
            ms_scaling_factor=ms_scaling_factor,
            schedule=schedule,
            omp_thread_count=omp_thread_count,
            osd_method=osd_method,
            osd_order=osd_order,
        )
        return bposd
        predictions = np.zeros((len(shots), dem.num_observables), dtype=bool)
        for i, shot in enumerate(shots):
            predictions[i, :] = (matrices.observables_matrix @ bposd.decode(shot)) % 2
        return predictions

def results():
    results = []

    for index, task in enumerate(generate_example_tasks()):
        circuit = task['circuit']

        
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(120, separate_observables=True)
        dem = circuit.detector_error_model(decompose_errors=True)

        bposd = decode(max_iter=10, 
                bp_method="ms", 
                ms_scaling_factor=0.625, 
                schedule="parallel", 
                omp_thread_count=1, 
                osd_method="osd0", 
                osd_order=0, 
                dem=dem, 
                shots=10_000)

        predictions = bposd.decode(detection_events)

        num_errors = 0
        for shot in range(10_000):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        return num_errors

        predictions = decode(max_iter=10, 
                bp_method="ms", 
                ms_scaling_factor=0.625, 
                schedule="parallel", 
                omp_thread_count=1, 
                osd_method="osd0", 
                osd_order=0, 
                dem=dem, 
                shots=shots)
    
        num_errors = np.sum(predictions, axis=0)
        logical_error_rate = num_errors / len(predictions)
        results.append({'p': task['p'], 'd': task['d'], 'rounds': task['rounds'], 'logical_error_rate': logical_error_rate})
    return results


def plots(results):
    fig, axis = plt.subplots(1, 1, figsize=(10, 5))
    for d in [5, 7, 9]:
        data = [result for result in results if result['d'] == d]
        ps = [result['p'] for result in data]
        error_rates = [result['logical_error_rate'] for result in data]
        axis.plot(ps, error_rates, label=f'd={d}')
    axis.set_ylabel('Logical Error Rate')
    axis.set_xlabel('Physical Error Rate')
    axis.set_title('BpOsdDecoder')
    axis.legend()
    plt.savefig("bposd_wo_sinter.jpg")
    plt.show()

plots(results())