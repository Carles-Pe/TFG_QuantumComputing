import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit.circuit.library.n_local.evolved_operator_ansatz import _is_pauli_identity

from qiskit.quantum_info import SparsePauliOp

class MultiangleQAOAAnsatz(EvolvedOperatorAnsatz):
    """A generalized QAOA quantum circuit with a support of custom initial states and mixers.

    References:

        [1]: Farhi et al., A Quantum Approximate Optimization Algorithm.
            `arXiv:1411.4028 <https://arxiv.org/pdf/1411.4028>`_
    """

    def __init__(
        self,
        cost_operator=None,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        mixer_operator=None,
        name: str = "QAOA",
        flatten: bool | None = None,
    ):
        r"""
        Args:
            cost_operator (BaseOperator or OperatorBase, optional): The operator
                representing the cost of the optimization problem, denoted as :math:`U(C, \gamma)`
                in the original paper. Must be set either in the constructor or via property setter.
            reps (int): The integer parameter p, which determines the depth of the circuit,
                as specified in the original paper, default is 1.
            initial_state (QuantumCircuit, optional): An optional initial state to use.
                If `None` is passed then a set of Hadamard gates is applied as an initial state
                to all qubits.
            mixer_operator (BaseOperator or OperatorBase or QuantumCircuit, optional): An optional
                custom mixer to use instead of the global X-rotations, denoted as :math:`U(B, \beta)`
                in the original paper. Can be an operator or an optionally parameterized quantum
                circuit.
            name (str): A name of the circuit, default 'qaoa'
            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple
                layers of gate objects. By default currently the contents of
                the output circuit will be wrapped in nested objects for
                cleaner visualization. However, if you're using this circuit
                for anything besides visualization its **strongly** recommended
                to set this flag to ``True`` to avoid a large performance
                overhead for parameter binding.
        """
        super().__init__(reps=reps, name=name, flatten=flatten)

        self._cost_operator = None
        self._reps = reps
        self._initial_state: QuantumCircuit | None = initial_state
        self._mixer = mixer_operator

        # set this circuit as a not-built circuit
        self._bounds: list[tuple[float | None, float | None]] | None = None

        # store cost operator and set the registers if the operator is not None
        self.cost_operator = cost_operator

        self.num_cost_terms = None
        self.num_mixer_terms = None

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True

        if not super()._check_configuration(raise_on_failure):
            return False

        if self.cost_operator is None:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    "The operator representing the cost of the optimization problem is not set"
                )

        if self.initial_state is not None and self.initial_state.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    "The number of qubits of the initial state {} does not match "
                    "the number of qubits of the cost operator {}".format(
                        self.initial_state.num_qubits, self.num_qubits
                    )
                )

        if self.mixer_operator is not None and self.mixer_operator.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    "The number of qubits of the mixer {} does not match "
                    "the number of qubits of the cost operator {}".format(
                        self.mixer_operator.num_qubits, self.num_qubits
                    )
                )

        return valid

    @property
    def parameter_bounds(self) -> list[tuple[float | None, float | None]] | None:
        """The parameter bounds for the unbound parameters in the circuit.

        Returns:
            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded
            parameter in the corresponding direction. If None is returned, problem is fully
            unbounded.
        """


        if self._bounds is not None:
            return self._bounds

        # if the mixer is a circuit, we set no bounds
        if isinstance(self.mixer_operator, QuantumCircuit):
            return None

        # default bounds: None for gamma (cost operator), [0, 2pi] for gamma (mixer operator)
        beta_bounds = (0, 2 * np.pi)
        gamma_bounds = (None, None)
        bounds: list[tuple[float | None, float | None]] = []

        if not _is_pauli_identity(self.mixer_operator):
            bounds += self.reps * [beta_bounds]

        if not _is_pauli_identity(self.cost_operator):
            bounds += self.reps * [gamma_bounds]

        return bounds

    @parameter_bounds.setter
    def parameter_bounds(self, bounds: list[tuple[float | None, float | None]] | None) -> None:
        """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
        self._bounds = bounds

    def expand_sparse_pauli_op(self, sparse_pauli_op):
        """Expand a SparsePauliOp into a list of individual SparsePauliOp terms."""
        # If it is not a SparsePauliOp, return a list with simply the operator
        if not isinstance(sparse_pauli_op, SparsePauliOp):
            return [sparse_pauli_op]
        
        pauli_terms = []
        for pauli, coeff in zip(sparse_pauli_op.paulis, sparse_pauli_op.coeffs):
            pauli_terms.append(SparsePauliOp([pauli], [coeff]))
        return pauli_terms

    @property
    def operators(self) -> list:
        """The operators that are evolved in this circuit.

        Returns:
             List[Union[BaseOperator, OperatorBase, QuantumCircuit]]: The operators to be evolved
                (and circuits) in this ansatz.
        """
        # Ensure the cost operator is a SparsePauliOp
        if not isinstance(self.cost_operator, SparsePauliOp):
            raise ValueError("The cost operator must be a SparsePauliOp")

        # Here, each operator is expanded individually so the parametrization is done in each term
        cost_operators = self.expand_sparse_pauli_op(self.cost_operator)
        mixer_operators = self.expand_sparse_pauli_op(self.mixer_operator)
        return cost_operators + mixer_operators

    @property
    def cost_operator(self):
        """Returns an operator representing the cost of the optimization problem.

        Returns:
            BaseOperator or OperatorBase: cost operator.
        """
        return self._cost_operator

    @cost_operator.setter
    def cost_operator(self, cost_operator) -> None:
        """Sets cost operator.

        Args:
            cost_operator (BaseOperator or OperatorBase, optional): cost operator to set.
        """
        self._cost_operator = cost_operator
        self.qregs = [QuantumRegister(self.num_qubits, name="q")]
        self._invalidate()

    @property
    def reps(self) -> int:
        """Returns the `reps` parameter, which determines the depth of the circuit."""
        return self._reps

    @reps.setter
    def reps(self, reps: int) -> None:
        """Sets the `reps` parameter."""
        self._reps = reps
        self._invalidate()

    @property
    def initial_state(self) -> QuantumCircuit | None:
        """Returns an optional initial state as a circuit"""
        if self._initial_state is not None:
            return self._initial_state

        # if no initial state is passed and we know the number of qubits, then initialize it.
        if self.num_qubits > 0:
            initial_state = QuantumCircuit(self.num_qubits)
            initial_state.h(range(self.num_qubits))
            return initial_state

        # otherwise we cannot provide a default
        return None

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit | None) -> None:
        """Sets initial state."""
        self._initial_state = initial_state
        self._invalidate()

    # we can't directly specify OperatorBase as a return type, it causes a circular import
    # and pylint objects if return type is not documented
    @property
    def mixer_operator(self):
        """Returns an optional mixer operator expressed as an operator or a quantum circuit.

        Returns:
            BaseOperator or OperatorBase or QuantumCircuit, optional: mixer operator or circuit.
        """
        if self._mixer is not None:
            return self._mixer

        # if no mixer is passed and we know the number of qubits, then initialize it.
        if self.cost_operator is not None:
            # local imports to avoid circular imports
            num_qubits = self.cost_operator.num_qubits

            # Mixer is just a sum of single qubit X's on each qubit. Evolving by this operator
            # will simply produce rx's on each qubit.
            mixer_terms = [
                ("I" * left + "X" + "I" * (num_qubits - left - 1), 1) for left in range(num_qubits)
            ]
            mixer = SparsePauliOp.from_list(mixer_terms)
            return mixer

        # otherwise we cannot provide a default
        return None

    @mixer_operator.setter
    def mixer_operator(self, mixer_operator) -> None:
        """Sets mixer operator.

        Args:
            mixer_operator (BaseOperator or OperatorBase or QuantumCircuit, optional): mixer
                operator or circuit to set.
        """
        self._mixer = mixer_operator
        self._invalidate()

    @property
    def num_qubits(self) -> int:
        if self._cost_operator is None:
            return 0
        return self._cost_operator.num_qubits

    def _build(self):
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        self.num_cost_terms = len(self.expand_sparse_pauli_op(self.cost_operator))
        self.num_mixer_terms = len(self.expand_sparse_pauli_op(self.mixer_operator))

        betas = ParameterVector("β", self.reps * self.num_mixer_terms)
        gammas = ParameterVector("γ", self.reps * self.num_cost_terms)

        # Create a permutation to take us from (cost_1_1, cost_1_2, ... , mixer_1_1, mixer_1_2, ... , cost_2_1, ... , mixer_2_1, ...)
        # to (cost_1_1, cost_1_2, ... , cost_2_1, ..., mixer_1_1, mixer_1_2, ... , mixer_2_1, ...), or if the mixer is a circuit
        reordered = []
        for rep in range(self.reps):
            reordered.extend(gammas[rep * self.num_cost_terms : (rep + 1) * self.num_cost_terms])
            reordered.extend(betas[rep * self.num_mixer_terms : (rep + 1) * self.num_mixer_terms])

        self.assign_parameters(dict(zip(self.ordered_parameters, reordered)), inplace=True)
