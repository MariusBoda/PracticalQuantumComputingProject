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

    #bp = BpDecoder(matrices.check_matrix,
    #    error_channel=list(matrices.priors),
    #    max_iter=max_iter,
    #    bp_method=bp_method,
    #    ms_scaling_factor=ms_scaling_factor
    #    )
    
    
    
    predictions = np.zeros((len(shots), dem.num_observables), dtype=bool)
    predictions = bposd.decode(shots)
    #for i, shot in enumerate(shots):
        #predictions = bposd.decode(shot)
        
    #for i, shot in enumerate(shots):
        #predictions[i, :] = (matrices.observables_matrix @ bposd.decode(shot).T) % 2
        #predictions[i, :] = (bposd.decode(shot).T @ matrices.observables_matrix) % 2
    return predictions
#obersvable_flips dim should equal matrices.observables_matrix @ bposd.decode(shot).T

def results():
    results = []
    num_shots = 10000
    for index, task in enumerate(generate_example_tasks()):
        print(f"Task # {index + 1}: p = {task['p']}, d = {task['d']}")
        circuit = task['circuit']

        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
        dem = circuit.detector_error_model(decompose_errors=True)

        predictions = decode(max_iter=10, 
                             bp_method="ms", 
                             ms_scaling_factor=0.625, 
                             schedule="parallel", 
                             omp_thread_count=1, 
                             osd_method="osd0", 
                             osd_order=0, 
                             dem=dem, 
                             shots=detection_events)

        num_errors = 0
        matrices = detector_error_model_to_check_matrices(dem)
        logical_num_errors = ((predictions @ matrices.observables_matrix.toarray().T) + observable_flips) % 2
        logical_num_errors = logical_num_errors.sum()
        print("logical num of errors: ", logical_num_errors)

        #for shot in range(num_shots):
         #   actual_for_shot = observable_flips[shot]
          #  predicted_for_shot = predictions[shot]
           # if not np.array_equal(actual_for_shot, predicted_for_shot):
            #    num_errors += 1


        logical_error_rate = logical_num_errors / num_shots
        #logical_error_rate = num_errors / num_shots
        ler_round = 1 - (1 - (logical_error_rate) ** (1/task['d']))
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
