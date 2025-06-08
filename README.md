# Quantum Teleportation Simulation

This project implements a quantum teleportation protocol using Qiskit, demonstrating the transfer of quantum states between qubits. The implementation includes both noiseless and noisy simulations, with visualization capabilities.

## Features

- Quantum teleportation circuit implementation
- Support for custom initial states
- Noise simulation with configurable error probability
- Visualization of results using Bloch sphere and histograms
- Comprehensive error handling

## Requirements

- Python 3.7+
- Qiskit
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

The main script can be run directly:

```bash
python quantum_teleportation.py
```

This will:
1. Create a superposition state (|+⟩)
2. Run the teleportation protocol both with and without noise
3. Generate visualizations saved as 'teleportation_results.png'

### Custom Usage

You can also import and use the functions in your own code:

```python
from quantum_teleportation import run_teleportation, visualize_results

# Define your initial state (must be normalized)
initial_state = [1/np.sqrt(2), 1/np.sqrt(2)]  # |+⟩ state

# Run teleportation
result_noiseless = run_teleportation(initial_state, use_noise=False)
result_noisy = run_teleportation(initial_state, use_noise=True, error_prob=0.01)

# Visualize results
visualize_results(initial_state, result_noiseless, result_noisy)
```

## Output

The script generates a visualization file 'teleportation_results.png' containing:
- Initial state on the Bloch sphere
- Measurement results for noiseless teleportation
- Measurement results for noisy teleportation

## Error Handling

The code includes validation for:
- Normalized quantum states
- Proper circuit initialization
- Backend compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details. 