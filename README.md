# BpOSD Decoder Simulation

## Overview

This project simulates quantum error correction using a BP-OSD (Belief Propagation with Ordered Statistics Decoding) decoder for a surface code. The simulation generates tasks to test different physical error rates, code distances, and OSD orders, then visualizes the results in the form of logical error rate plots.

## Features

- **Task Generation**: Creates tasks for different physical error rates (`p` values) and code distances (`d` values). These tasks simulate different levels of noise and quantum error-correcting code strength.
- **Decoder Simulation**: Runs simulations using the BP-OSD decoder, allowing customization of the OSD method and order. The decoder tests the logical error rates against varying parameters.
- **Error Model**: Uses `stim` to generate a surface code circuit and applies detection events and observable flips for the quantum error correction tasks.
- **Results Visualization**: Plots the logical error rate as a function of the physical error rate, with different curves for varying code distances and OSD orders.

## Workflow

1. **Task Generation**: The program first generates example tasks, which vary by:
   - Physical error rate (`p`)
   - Code distance (`d`)
   - Number of rounds (`rounds`)
   
2. **Decoding**: For each task, the `BpOsdDecoder` is applied using the provided circuit data. This simulates the performance of quantum error correction and calculates the logical error rate.

3. **Data Logging**: Simulation results are saved to CSV files to allow reloading data without rerunning the simulations.

4. **Visualization**: Plots are generated showing the logical error rates for various configurations of the OSD method and order. The x-axis represents the physical error rate, and the y-axis shows the logical error rate on a log-log scale.

## Input Parameters

- `decoder_type`: Specifies the type of decoder used (default is `BpOsd`).
- `osd_method`: Chooses the OSD method for decoding (e.g., `osd_E`).
- `osd_order`: Specifies the OSD order used in the decoding process (e.g., 0, 5, 15).
- `load_from_file`: A boolean parameter to either load precomputed results from a CSV file or run the simulation from scratch.

## Output

- **CSV Files**: The simulation results, including the physical error rate, code distance, rounds, and logical error rates, are saved to CSV files.
- **Plots**: Logical error rate vs physical error rate plots are generated for each OSD method and order, with different curves for different code distances.

## Usage

To run the simulation and generate plots, simply execute the script. You can modify the OSD methods or orders and decide whether to load previous results from CSV or run the simulation from scratch.

