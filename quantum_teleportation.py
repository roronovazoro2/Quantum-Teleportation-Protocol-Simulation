import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import depolarizing_error
from qiskit.primitives import Sampler
import matplotlib.pyplot as plt

def validate_state(state):
    """
    Validates that the input state is normalized.
    Args:
        state: The quantum state to validate
    Returns:
        bool: True if state is valid, False otherwise
    """
    if state is None:
        return True
    norm = np.sqrt(np.sum(np.abs(state)**2))
    return np.isclose(norm, 1.0, atol=1e-10)

def create_teleportation_circuit(initial_state=None):
    """
    Creates a quantum circuit for teleportation protocol.
    Args:
        initial_state: The state to be teleported (if None, uses |0⟩)
    Returns:
        QuantumCircuit: The teleportation circuit
    Raises:
        ValueError: If initial_state is not normalized
    """
    if not validate_state(initial_state):
        raise ValueError("Initial state must be normalized")
        
    # Create quantum registers
    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # Prepare initial state if provided
    if initial_state is not None:
        circuit.initialize(initial_state, 0)
    
    # Create Bell pair between qubits 1 and 2
    circuit.h(1)
    circuit.cx(1, 2)
    
    # Bell measurement
    circuit.cx(0, 1)
    circuit.h(0)
    circuit.measure([0, 1], [0, 1])
    
    # Apply corrections based on measurement results
    circuit.x(2).c_if(cr[1], 1)
    circuit.z(2).c_if(cr[0], 1)
    
    return circuit

def add_noise(circuit, error_prob=0.01):
    """
    Adds depolarizing noise to the circuit.
    Args:
        circuit: The quantum circuit
        error_prob: Probability of error
    Returns:
        tuple: (noisy_circuit, noise_model)
    """
    noise_model = NoiseModel()
    error = depolarizing_error(error_prob, 1)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])
    return circuit, noise_model

def run_teleportation(initial_state=None, use_noise=False, error_prob=0.01):
    """
    Runs the teleportation protocol with optional noise.
    Args:
        initial_state: The state to be teleported
        use_noise: Whether to add noise to the circuit
        error_prob: Probability of error if using noise
    Returns:
        dict: Results of the simulation
    """
    circuit = create_teleportation_circuit(initial_state)
    
    if use_noise:
        circuit, noise_model = add_noise(circuit, error_prob)
        backend = Aer.get_backend('qasm_simulator')
        job = backend.run(circuit, shots=1000, noise_model=noise_model)
    else:
        backend = Aer.get_backend('statevector_simulator')
        job = backend.run(circuit, shots=1000)
    
    result = job.result()
    return result

def visualize_results(initial_state, result_noiseless, result_noisy):
    """
    Creates visualizations of the teleportation results.
    Args:
        initial_state: The initial state that was teleported
        result_noiseless: Results from noiseless simulation
        result_noisy: Results from noisy simulation
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot initial state on Bloch sphere
    if initial_state is not None:
        plot_bloch_multivector(initial_state, ax=ax1)
        ax1.set_title('Initial State')
    
    # Plot measurement results
    if hasattr(result_noiseless, 'get_counts'):
        plot_histogram(result_noiseless.get_counts(), ax=ax2)
        ax2.set_title('Noiseless Results')
    
    if hasattr(result_noisy, 'get_counts'):
        plot_histogram(result_noisy.get_counts(), ax=ax3)
        ax3.set_title('Noisy Results')
    
    plt.tight_layout()
    plt.savefig('teleportation_results.png')
    plt.close()

def main():
    # Example: Teleport a superposition state
    initial_state = [1/np.sqrt(2), 1/np.sqrt(2)]  # |+⟩ state
    
    # Run teleportation without noise
    result_noiseless = run_teleportation(initial_state, use_noise=False)
    
    # Run teleportation with noise
    result_noisy = run_teleportation(initial_state, use_noise=True, error_prob=0.01)
    
    # Visualize results
    visualize_results(initial_state, result_noiseless, result_noisy)
    
    print("Teleportation simulation completed. Results saved to 'teleportation_results.png'")

if __name__ == "__main__":
    main() 