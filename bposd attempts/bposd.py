import stim
import numpy as np
from ldpc.bposd_decoder import BpOsdDecoder
import matplotlib.pyplot as plt
from beliefmatching import detector_error_model_to_check_matrices


class BpOsdDecoderWrapper:
    def __init__(self, max_iter=10, bp_method="ms", ms_scaling_factor=0.625, schedule="parallel", omp_thread_count=1, osd_method="osd0", osd_order=0):
        self.max_iter = max_iter
        self.bp_method = bp_method
        self.ms_scaling_factor = ms_scaling_factor
        self.schedule = schedule
        self.omp_thread_count = omp_thread_count
        self.osd_method = osd_method
        self.osd_order = osd_order

    def decode(self, dem, dets_b8):
        matrices = detector_error_model_to_check_matrices(dem)
        bposd = BpOsdDecoder(
            matrices.check_matrix,
            error_channel=list(matrices.priors),
            max_iter=self.max_iter,
            bp_method=self.bp_method,
            ms_scaling_factor=self.ms_scaling_factor,
            schedule=self.schedule,
            omp_thread_count=self.omp_thread_count,
            osd_method=self.osd_method,
            osd_order=self.osd_order,
        )
        predictions = np.zeros((len(dets_b8), dem.num_observables), dtype=bool)
        for i, shot in enumerate(dets_b8):
            predictions[i, :] = (matrices.observables_matrix @ bposd.decode(shot)) % 2
        return predictions


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


def main():
    decoder = BpOsdDecoderWrapper()
    results = []

    print("Starting task generation and decoding...")

    for i, task in enumerate(generate_example_tasks(), 1):
        print(f"\nProcessing task {i}...")
        circuit = task['circuit']
        print(f"Generated circuit with p={task['p']}, d={task['d']}, rounds={task['rounds']}")
        dem = circuit.detector_error_model(decompose_errors=True)
        shots = circuit.compile_detector_sampler().sample(1000)
        predictions = decoder.decode(dem, shots)
        num_errors = np.sum(predictions, axis=0)
        logical_error_rate = num_errors / len(predictions)
        results.append({'p': task['p'], 'd': task['d'], 'rounds': task['rounds'], 'logical_error_rate': logical_error_rate})
        print(f"Task {i} completed. Logical error rate: {logical_error_rate}")

    print("\nAll tasks completed. Generating plot...")

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

    fig.savefig('plot.png')
    plt.show()

    print("Plot saved as 'plot.png' and displayed.")


if __name__ == '__main__':
    main()
