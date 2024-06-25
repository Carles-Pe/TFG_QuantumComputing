from .MultiangleQAOAAnsatz import MultiangleQAOAAnsatz

import numpy as np
from typing import Any, Dict, List, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Pauli, SparsePauliOp


class TFG_QAOA():
    """
    A class that relies on the QAOAAnsatz implementation for most of its cases but
    allows leeway for custom implementations
    """
    def __init__(self,
                 cost_operator=None,
                 reps: int = 1,
                 initial_state=None,
                 mixer_operator=None,
                 qaoa_options: dict = None
                 ) -> None:
        """
        Initialize the TFG_QAOA class.

        Parameters:
        cost_operator (PauliOp): The cost operator for the QAOA.
        reps (int): The number of repetitions of the QAOA circuit.
        initial_state (QuantumCircuit): The initial state for the QAOA circuit.
        mixer_operator (QuantumCircuit): The mixer operator for the QAOA circuit.
        qaoa_options (dict): Additional options for the QAOA.
        """
        
        if cost_operator is None:
            raise ValueError("Cost operator must be defined")
        
        self.cost_operator = cost_operator
        self.nqubits = cost_operator.num_qubits
        self.reps = reps
        self.initial_state = initial_state
        self.mixer_operator = mixer_operator


        if qaoa_options is None:
            self.qaoa_options = {
                'parametrization': None,
                'warm_start': None,
                'multiangle': False,
            }
        else:
            # Make sure parametrization is valid [None, 'fourier']
            if qaoa_options['parametrization'] not in [None, 'fourier']:
                raise ValueError("Invalid parametrization option")

            self.qaoa_options = qaoa_options


        self.cost_function = None

        if self.qaoa_options['parametrization'] == 'fourier':
            self.cost_function = self.cost_function_fourier
        elif self.qaoa_options['parametrization'] is None:
            self.cost_function = self.cost_function_basic
        else:
            raise ValueError("Invalid parametrization option")

        self.qaoa_ansatz = None
        self.num_parameters = -1

        self._build_qaoa_ansatz(self.reps)

        self.num_cost_terms = self.qaoa_ansatz.num_cost_terms if self.qaoa_options['multiangle'] else 1
        self.num_mixer_terms = self.qaoa_ansatz.num_mixer_terms if self.qaoa_options['multiangle'] else 1
        if self.qaoa_options['warm_start'] is not None:
            self.num_mixer_terms = 1

        return

    def build_qaoa_ansatz(self, p: int):
        """
        Build the QAOA ansatz and return it, without changing the underlying object.

        Parameters:
        p (int): The number of repetitions of the QAOA circuit.
        """
        return self._build_qaoa_ansatz(p, inplace=False)
    
    def _build_qaoa_ansatz(self, p: int, inplace: bool = True):
        """
        Internal method to build the QAOA ansatz.

        Parameters:
        p (int): The number of repetitions of the QAOA circuit.
        inplace (bool): Whether to build the ansatz in place or return it.

        Returns:
        QAOAAnsatz: The constructed QAOA ansatz if inplace is False.
        """

        
        # Now we generate the ansatz per se:
        the_initial_state = self.initial_state
        the_mixer_operator = self.mixer_operator

        if self.qaoa_options['warm_start'] is not None:
            # We have been given a warm-started classical solution, so it will redifine the initial state and
            # the mixer operator

            # warm_start values = [chi_1, chi_2, ... , chi_n]

            the_initial_state = self._build_warm_start_initial_state(self.qaoa_options['warm_start'], inplace)
            the_mixer_operator = self._build_warm_start_mixer_operator(self.qaoa_options['warm_start'], inplace)

        
        if self.qaoa_options['multiangle']:
            if inplace:
                self.qaoa_ansatz = MultiangleQAOAAnsatz(cost_operator=self.cost_operator, 
                                            reps=p, 
                                            initial_state=self.initial_state, 
                                            mixer_operator=self.mixer_operator)
                self.num_parameters = self.qaoa_ansatz.num_parameters
            else:
                return MultiangleQAOAAnsatz(cost_operator=self.cost_operator, 
                                reps=p, 
                                initial_state=the_initial_state, 
                                mixer_operator=the_mixer_operator)

        else:
            if inplace:
                self.qaoa_ansatz = QAOAAnsatz(cost_operator=self.cost_operator, 
                                            reps=p, 
                                            initial_state=self.initial_state, 
                                            mixer_operator=self.mixer_operator)
                self.num_parameters = self.qaoa_ansatz.num_parameters
            else :
                return QAOAAnsatz(cost_operator=self.cost_operator, 
                                reps=p, 
                                initial_state=the_initial_state, 
                                mixer_operator=the_mixer_operator)


    def _build_warm_start_initial_state(self, chis: list, inplace: bool = True):
        """
        Build the initial state for the QAOA from warm-start values.

        Parameters:
        chis (list): The warm-start values.
        """

        theta_list = [2*np.arcsin(np.sqrt(chi_i)) for chi_i in chis]

        initial_state = QuantumCircuit(self.nqubits)
        for qubit, theta_i in enumerate(theta_list):
            initial_state.ry(theta_i, qubit)

        if inplace:
            self.initial_state = initial_state
        else:
            return initial_state
    

    def _build_warm_start_mixer_operator(self, chis: list, inplace: bool = True):
        """
        Build the mixer operator for the QAOA from warm-start values.

        Parameters:
        chis (list): The warm-start values.
        """

        theta_list = [2*np.arcsin(np.sqrt(chi_i)) for chi_i in chis]

        # Rememeber that the order in which operators are applied is right to left
        beta = Parameter('β')

        mixer_operator = QuantumCircuit(self.nqubits)
        for qubit, theta_i in enumerate(theta_list):
            mixer_operator.ry(-theta_i, qubit)
            mixer_operator.rz(-2*beta, qubit)
            mixer_operator.ry(theta_i, qubit)

        if inplace:
            self.mixer_operator = mixer_operator
        else:
            return mixer_operator

    
    def cost_function_basic(self, params, ansatz, hamiltonian, estimator):
        """Return estimate of energy from estimator

        Parameters:
            params (ndarray): Array of ansatz parameters
            ansatz (QuantumCircuit): Parameterized ansatz circuit
            hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
            estimator (Estimator): Estimator primitive instance

        Returns:
            float: Energy estimate
        """
        cost = estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
        return cost
    

    def cost_function_fourier(self, params, ansatz, hamiltonian, estimator):
        """
        Return estimate of energy from estimator using Fourier parametrization.
        Parameters must be inserted in the following way (alphabetical order):
        ParameterView([ParameterVectorElement(β[0]), ParameterVectorElement(β[1]), ParameterVectorElement(γ[0]), ParameterVectorElement(γ[1])])


        Parameters:
        params (ndarray): Array of ansatz parameters as  [u1, u2, u3, u4, ... , uq, v1, v2, v3, v4, ... , vq].
        ansatz (QuantumCircuit): Parameterized ansatz circuit.
        hamiltonian (PauliOp): Operator representation of Hamiltonian.
        estimator (Estimator): Estimator primitive instance.

        Returns:
        float: Energy estimate.
        """
        if self.qaoa_options['multiangle']:
            return ValueError("Fourier parametrization is not supported for multiangle QAOA, yet...")
        
        q = len(params) // 2
        gamma = self.get_gammas_fourier(params[:q], ansatz.reps)
        beta = self.get_betas_fourier(params[q:], ansatz.reps)


        cost = estimator.run(ansatz, hamiltonian, parameter_values=np.concatenate((beta,gamma),axis=0)).result().values[0]
        return cost
    

    def get_gammas_fourier(self, params, p):
        """
        Compute gamma parameters using Fourier series.
        Using Fourier parameters described in:
        https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.021067

        Parameters:
        params (list): Fourier parameters as [u1, u2, u3, u4, ... , uq].
        p (int): Number of repetitions.

        Returns:
        ndarray: Gamma parameters.
        """
        q = len(params)
        gamma = np.zeros(p)
        for i in range(p):
            gamma[i] = sum(params[k] * np.sin((k - 0.5) * (i - 0.5) * np.pi / p) for k in range(q))

        return gamma
    
    def get_betas_fourier(self, params, p):
        """
        Compute beta parameters using Fourier series.
        Using Fourier parameters described in:
        https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.021067

        Parameters:
        params (list): Fourier parameters as [v1, v2, v3, v4, ... , vq].
        p (int): Number of repetitions.

        Returns:
        ndarray: Beta parameters.
        """

        q = len(params)
        beta = np.zeros(p)
        for i in range(p):
            beta[i] = sum(params[k] * np.cos((k - 0.5) * (i - 0.5) * np.pi / p) for k in range(q))

        return beta    


class QAOARun:
    """
    A class to store the results of a QAOA execution.

    Attributes:
        initial_params (np.ndarray): The initial parameters used for the QAOA run.
        qaoa_config (Dict): The QAOA configuration options.
        model_name (str): The name of the model used.
        num_iterations (int): The number of iterations executed by the optimizer.
        final_params (np.ndarray): The final parameters obtained after optimization.
        optimizer (str): The optimizer used for the QAOA run.
        final_value (float): The final value of the optimized function.
        elasped_time (List[float]): The time taken for each level of p. 
        cost_operator (PauliOp): The cost operator used in the QAOA.
        mixer_operator (QuantumCircuit): The mixer operator used in the QAOA.
        initial_state (QuantumCircuit): The initial state used in the QAOA.
    """

    def __init__(self,
                 initial_params: np.ndarray,
                 qaoa_config: Dict[str, Any],
                 model_name: str,
                 reps: int,
                 num_iterations: int,
                 final_params: np.ndarray,
                 optimizer: str,
                 final_value: float,
                 elasped_time: List[float],
                 cost_operator: Any,
                 mixer_operator: Optional[Any] = None,
                 initial_state: Optional[Any] = None) -> None:
        self.initial_params = initial_params
        self.qaoa_config = qaoa_config
        self.model_name = model_name
        self.reps = reps
        self.num_iterations = num_iterations
        self.final_params = final_params
        self.optimizer = optimizer
        self.final_value = final_value
        self.elasped_time = elasped_time
        self.cost_operator = cost_operator
        self.mixer_operator = mixer_operator
        self.initial_state = initial_state


    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the QAOARun instance to a dictionary.

        Returns:
        Dict[str, Any]: A dictionary representation of the QAOARun instance.
        """
        def complex_to_tuple(x):
            return (x.real, x.imag) if isinstance(x, complex) else x
        
        def serialize_pauli_op(pauli_op):
            return {
                'paulis': pauli_op.paulis.to_labels(),
                'coefficients': [complex_to_tuple(coeff) for coeff in pauli_op.coeffs]
            }
        return {
            'initial_params': self.initial_params.tolist(),
            'qaoa_config': self.qaoa_config,
            'model_name': self.model_name,
            'reps': self.reps,
            'num_iterations': self.num_iterations,
            'final_params': self.final_params.tolist(),
            'optimizer': self.optimizer,
            'final_value': self.final_value,
            'elasped_time': self.elasped_time,
            'cost_operator': serialize_pauli_op(self.cost_operator) if self.cost_operator else None,
            'mixer_operator': self.mixer_operator.qasm() if self.mixer_operator else None,
            'initial_state': self.initial_state.qasm() if self.initial_state else None
        }

    def from_dict(cls, data: Dict[str, Any]) -> 'QAOARun':
        """
        Create a QAOARun instance from a dictionary.

        Args:
        data (Dict[str, Any]): A dictionary containing the QAOARun data.

        Returns:
        QAOARun: A QAOARun instance created from the dictionary.
        """
        initial_params = np.array(data['initial_params'])
        final_params = np.array(data['final_params'])
        cost_operator = SparsePauliOp.from_list(data['cost_operator']) if data['cost_operator'] else None
        mixer_operator = QuantumCircuit.from_qasm_str(data['mixer_operator']) if data['mixer_operator'] else None
        initial_state = QuantumCircuit.from_qasm_str(data['initial_state']) if data['initial_state'] else None

        return cls(
            initial_params=initial_params,
            qaoa_config=data['qaoa_config'],
            model_name=data['model_name'],
            reps=data['reps'],
            num_iterations=data['num_iterations'],
            final_params=final_params,
            optimizer=data['optimizer'],
            final_value=data['final_value'],
            elasped_time=data['elasped_time'],
            cost_operator=cost_operator,
            mixer_operator=mixer_operator,
            initial_state=initial_state
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the QAOA run.

        Returns:
        Dict[str, Any]: A dictionary containing the summary of the QAOA run.
        """
        summary = {
            'initial_params': self.initial_params,
            'qaoa_config': self.qaoa_config,
            'model_name': self.model_name,
            'reps': self.reps,
            'num_iterations': self.num_iterations,
            'final_params': self.final_params,
            'optimizer': self.optimizer,
            'final_value': self.final_value,
            'elasped_time': self.elasped_time,
            'cost_operator': self.cost_operator,
            'mixer_operator': self.mixer_operator,
            'initial_state': self.initial_state
        }
        return summary

    def __str__(self) -> str:
        """
        String representation of the QAOA run.

        Returns:
        str: A string summarizing the QAOA run.
        """
        summary = self.get_summary()
        summary_str = "\n".join([f"{key}: {value}" for key, value in summary.items()])
        return f"QAOARun Summary:\n{summary_str}"
