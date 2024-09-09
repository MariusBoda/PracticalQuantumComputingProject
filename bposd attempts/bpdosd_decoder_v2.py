import stim
import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.bp_decoder import BpDecoder
import matplotlib.pyplot as plt
from beliefmatching import detector_error_model_to_check_matrices

def generate_example_tasks():
    for p in np.arange(0.001, 0.01, 0.002):
        for d in [3, 5]:
            sc_circuit = stim.Circuit.generated(
                rounds=d,
                distance=d,
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p,
                before_round_data_depolarization=p,
                code_task=f'surface_code:rotated_memory_z',
            )
            yield {'circuit': sc_circuit, 'p': p, 'd': d, 'rounds': d}

def decode(max_iter, 
           bp_method, 
           ms_scaling_factor, 
           schedule, 
           omp_thread_count, 
           osd_method, 
           osd_order, 
           matrices, shots):
    
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
    
    predictions = np.zeros((len(shots), matrices.check_matrix.shape[1]), dtype=bool)
    for i in range(len(shots)):
        predictions[i, :] = bposd.decode(shots[i, :])
    return predictions

def results():
    results = []
    num_shots = 10000
    for index, task in enumerate(generate_example_tasks()):
        print(f"Task # {index + 1}: p = {task['p']}, d = {task['d']}")
        circuit = task['circuit']

        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
        dem = circuit.detector_error_model(decompose_errors=True)
        matrices = detector_error_model_to_check_matrices(dem)

        predictions = decode(max_iter=10, 
                             bp_method="ms", 
                             ms_scaling_factor=0.625, 
                             schedule="parallel", 
                             omp_thread_count=1, 
                             osd_method="osd0", 
                             osd_order=0, 
                             matrices=matrices, 
                             shots=detection_events)

        logical_num_errors = ((predictions @ matrices.observables_matrix.toarray().T) + observable_flips) % 2

        logical_num_errors = logical_num_errors.sum()
        logical_error_rate = logical_num_errors / num_shots
        ler_round = 1 - (1 - ((logical_error_rate) ** (1/task['d']) ))

        results.append({'p': task['p'], 'd': task['d'], 'rounds': task['rounds'], 'logical_error_rate': ler_round})
    
    return results

def plots(results):
    fig, axis = plt.subplots(1, 1, figsize=(10, 5))
    for d in [3, 5]:
        data = [result for result in results if result['d'] == d]
        ps = [result['p'] for result in data]
        error_rates = [result['logical_error_rate'] for result in data]
        axis.plot(ps, error_rates, label=f'd={d}')
    axis.set_ylabel('Logical Error Rate')
    axis.set_xlabel('Physical Error Rate')
    axis.set_title('BpOsdDecoder')
    axis.legend()
    axis.loglog()
    #plt.savefig("bposd_wo_sinter.jpg")
    plt.show()

plots(results())

