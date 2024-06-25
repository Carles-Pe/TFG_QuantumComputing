from .qaoa import QAOARun
from .qaoa import TFG_QAOA
from .solver import TFG_Solver

import pickle
from typing import List, Dict

from qiskit.quantum_info import SparsePauliOp



def historyOfRuns2Pickle(file_path: str, data: List[Dict[str, List[QAOARun]]]) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def pickleHistoryOfRuns(file_path: str) -> List[Dict[str, List[QAOARun]]]:
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    

def construct_qaoa_and_solver_from_QAOARun(qaoa_run: QAOARun):

    qaoa = TFG_QAOA(cost_operator=qaoa_run.cost_operator,
                    mixer_operator=qaoa_run.mixer_operator,
                    initial_state=qaoa_run.initial_state,
                    reps=qaoa_run.reps,
                    qaoa_options=qaoa_run.qaoa_config)
    
    opt_opts = {
         'name': qaoa_run.optimizer,
         'disp': False
    }
    
    solver = TFG_Solver(optimizer_options=opt_opts)

    return qaoa, solver


def expand_sparse_pauli_op(self, sparse_pauli_op):
        """Expand a SparsePauliOp into a list of individual SparsePauliOp terms."""
        pauli_terms = []
        for pauli, coeff in zip(sparse_pauli_op.paulis, sparse_pauli_op.coeffs):
            pauli_terms.append(SparsePauliOp([pauli], [coeff]))
        return pauli_terms