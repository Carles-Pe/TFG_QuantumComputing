from .qaoa import TFG_QAOA, QAOARun
from .models import TFG_QuadraticModel
from .MultiangleQAOAAnsatz import MultiangleQAOAAnsatz

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union
import time
import copy
from enum import Enum

from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import Sampler as LocalSampler
from qiskit.primitives import Estimator as LocalEstimator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.problems.variable import VarType
from qiskit_optimization.algorithms import CplexOptimizer

# SciPy minimizer routine
from scipy.optimize import minimize


class TFG_Solver():
    def __init__(self,
                 initial_parameter_guess: np.ndarray = None,
                sampler_options: dict = None,
                estimator_options: dict = None,
                optimizer_options: dict = None,) -> None:
        
        if sampler_options is None:
            self.sampler_options = {
                'execution': 'local',
                'shots': 100,
            }
        else:
            # Check the execution is either 'local' or 'runtime'
            if sampler_options['execution'] not in ['local', 'runtime']:
                raise ValueError("Invalid execution option")

            self.sampler_options = sampler_options


        if estimator_options is None:
            self.estimator_options = {
                'execution': 'local',
                'shots': 100,
            }
        else:
            # Check the execution is either 'local' or 'runtime'
            if estimator_options['execution'] not in ['local', 'runtime']:
                raise ValueError("Invalid execution option")

            self.estimator_options = estimator_options


        if optimizer_options is None:
            self.optimizer_options = {
                'name': 'COBYLA',
                'disp': False,
            }
        else:
            # Check the optimizer is a valid one
            # Must be one of:
            # - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
            # - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
            # - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
            # - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
            # - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
            # - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
            # - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
            # - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
            # - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
            # - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
            # - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
            # - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
            # - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
            # - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`

            # Extracted form scipy.optimize.minimize
            if optimizer_options['name'] not in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']:
                raise ValueError("Invalid optimizer option")

            self.optimizer_options = optimizer_options

        # Build this objects based on the options
        self.local_backend = AerSimulator()
        self.runtime_backend = None
        self.sampler = None
        self.estimator = None

        self._build_sampler()
        self._build_estimator()
        return

    
    def relax_problem(self,problem: TFG_QuadraticModel) -> QuadraticProgram:
        """Change all variables to continuous."""
        relaxed_problem = copy.deepcopy(problem)
        for variable in relaxed_problem.variables:
            variable.vartype = VarType.CONTINUOUS

        return relaxed_problem
    
    def CPLEX_solver(self, problem: TFG_QuadraticModel):
        relaxed_qubo = self.relax_problem(problem.CPLEX_model)
        relaxed_solution = CplexOptimizer().solve(relaxed_qubo)

        return [relaxed_solution.x[i] for i in range(problem.nqubits)]
    
    def create_hmi_ws(self, chi_i):
        sqrt_1_chi = np.sqrt(1 - chi_i)
        sqrt_chi = np.sqrt(chi_i)
        U = np.array([[sqrt_1_chi, sqrt_chi],
                    [-sqrt_chi, sqrt_1_chi]])
        Z = np.array([[-1, 0],
                    [0, 1]])
        U_dagger = np.array([[sqrt_1_chi, -sqrt_chi],
                            [sqrt_chi, sqrt_1_chi]])
        H_M_i_ws = U_dagger @ Z @ U
        return Operator(H_M_i_ws)

    def create_hm_ws(self,chi_values):
        n = len(chi_values)  # Number of qubits
        operators = [self.create_hmi_ws(chi) for chi in chi_values]
        
        # Compute the tensor product of all individual qubit operators
        full_operator = operators[0]
        for op in operators[1:]:
            full_operator = full_operator.tensor(op)
        
        return Operator(full_operator)
    


    ### When given a QAOA #####################################################################################
    ###########################################################################################################
    def evaluate_cost(self, qaoa: TFG_QAOA, params):
        return qaoa.cost_function(params, qaoa.qaoa_ansatz, qaoa.cost_operator, self.estimator)

    def run_job(self, qaoa: TFG_QAOA, anstatz: Union[QAOAAnsatz,MultiangleQAOAAnsatz], params):
        # This function selects the behaviour being used, remote execution in IBM Quantum Computer or local simulation
        if self.estimator_options['execution'] == 'local':
            return minimize(qaoa.cost_function, 
                            params, 
                            method=self.optimizer_options['name'], 
                            args=(anstatz, 
                                    qaoa.cost_operator, 
                                    self.estimator),
                            options={'disp': self.optimizer_options['disp']})
        elif self.estimator_options['execution'] == 'runtime':
            raise ValueError("Runtime execution not implemented")
        else:
            raise ValueError("Invalid execution option")


    def check_commutability(self, qaoa: TFG_QAOA):
        # Check if the cost and mixer operators commute
        h_p_matrix = qaoa.cost_operator.to_operator()

        chi_values = qaoa.qaoa_options['warm_start']
        if chi_values is None:
            h_m_matrix = Operator(qaoa.mixer_operator)
        else:
            h_m_matrix = self.create_hm_ws(chi_values)

        commutator = h_p_matrix @ h_m_matrix - h_m_matrix @ h_p_matrix

        # If the commutator is zero, then the matrices commute
        n = qaoa.cost_operator.num_qubits
        return np.allclose(commutator.data, np.zeros((2**n, 2**n)))



    def optimize(self,
                 qaoa: TFG_QAOA,
                 initial_p1_guess: np.ndarray = None,
                 num_runs_at_p1: int = 10,
                 num_runs_at_each_p_increment: int = 1
        ):
        # TODO: Logic on how to get the first parameters
        # if qaoa.qaoa_options['multiangle']:
        #     raise ValueError("Not implemented for multiangle option")

        min_E, p1_params, elapsed_time_p1 = self.optimize_p1(qaoa, runs = num_runs_at_p1, initial_point = initial_p1_guess)
        print("At p=1, params optimal are ", p1_params, " with energy ", min_E)
        # _, p1_params = self.cost_landscape_p1(qaoa)

        # Now we optimize the rest of the parameters
        params = np.array(p1_params)
        current_depth = 1
        # Then execute minimization routine starting with the optimal parameters for depth 1

        best_run = {
            "best_p": 1,
            "best_params": params,
            "best_energy": min_E
        }

        qaoa_runs = []

        if qaoa.reps == 1:
            qaoa_run = QAOARun(
                initial_params=initial_p1_guess,
                qaoa_config=qaoa.qaoa_options,
                model_name="QAOA_Model",
                reps=1,
                num_iterations=-1,
                final_params=p1_params,
                optimizer=self.optimizer_options['name'],
                final_value=min_E,
                elasped_time=[elapsed_time_p1],
                cost_operator=qaoa.cost_operator,
                mixer_operator=qaoa.mixer_operator,
                initial_state=qaoa.initial_state
            )
            qaoa_runs.append(qaoa_run)
            return best_run, qaoa_runs
        
        #print("Current depth and length of params", current_depth, len(params))

        elapsed_times = [elapsed_time_p1]

        for current_depth in range(2, qaoa.reps+1):
            # Obtain the optimal parameters for the current depth
            best_intermediate_energy = np.inf
            best_intermediate_params = None
            intermediate_qaoa = qaoa.build_qaoa_ansatz(current_depth)

            for i in range(num_runs_at_each_p_increment):
                params = np.concatenate((params , [0] * qaoa.num_cost_terms + [0] * qaoa.num_mixer_terms),axis=0)
                print("Current depth and length of params", current_depth, len(params))

                # Measure time taken
                start_time = time.time()


                result_optimization = self.run_job(qaoa, intermediate_qaoa, params)
                
                elapsed_time = time.time() - start_time
                elapsed_times.append(elapsed_time)

                # print(f"Optimal parameters for depth {current_depth}: {result_optimization.x}, and yield energy: {result_optimization.fun}")

                if result_optimization.fun < best_run["best_energy"]:
                    best_run["best_p"] = current_depth
                    best_run["best_params"] = result_optimization.x
                    best_run["best_energy"] = result_optimization.fun

                if result_optimization.fun < best_intermediate_energy:
                    best_intermediate_energy = result_optimization.fun
                    best_intermediate_params = result_optimization.x

            # Update the optimal parameters for the next depth

            qaoa_run = QAOARun(
                initial_params=params,
                qaoa_config=qaoa.qaoa_options,
                model_name="QAOA_Model",
                reps=current_depth,
                num_iterations=-1,
                final_params=best_intermediate_params,
                optimizer=self.optimizer_options['name'],
                final_value=best_intermediate_energy,
                elasped_time=elapsed_times,
                cost_operator=qaoa.cost_operator,
                mixer_operator=qaoa.mixer_operator,
                initial_state=qaoa.initial_state
            )
            qaoa_runs.append(qaoa_run)

            params = best_intermediate_params

        self.opt_params = params
        self.last_result = result_optimization


        return best_run, qaoa_runs

    def optimize_with_only_start_num_runs(self, 
                                          qaoa: TFG_QAOA, 
                                          num_runs: int = 50, 
                                          initial_point_strategy: str = 'uniform'):
        """
        Optimize the QAOA using multiple random initial points for statistical analysis.

        Parameters:
        qaoa (TFG_QAOA): The QAOA instance to optimize.
        num_runs (int): Number of different random initial points.
        initial_point_strategy (str): Strategy for generating initial points ('uniform', 'perturbed', 'gaussian').

        Returns:
        List[QAOARun]: List of QAOARun instances for each run.
        """
        qaoa_runs = []
        
        for run in range(num_runs):
            if initial_point_strategy == 'uniform':
                initial_point = np.random.uniform(0, 2 * np.pi, size=2)
            elif initial_point_strategy == 'perturbed':
                base_point = np.random.uniform(0, 2 * np.pi, size=2)
                perturbation = np.random.normal(0, 0.1, size=2)
                initial_point = base_point + perturbation
            elif initial_point_strategy == 'gaussian':
                mean_point = np.full(2, np.pi)
                initial_point = np.random.normal(mean_point, 0.1, size=2)
            else:
                raise ValueError("Invalid initial point strategy")

            best_run, run_data = self.optimize(qaoa, 
                                               initial_p1_guess=initial_point, 
                                               num_runs_at_p1=1, 
                                               num_runs_at_each_p_increment=1)
            qaoa_runs.extend(run_data)
        
        return qaoa_runs
    

    def straight_minimization(self, 
                            qaoa: TFG_QAOA, 
                            num_runs: int = 50, 
                            initial_point_strategy: str = 'uniform'):
        """
        Optimize the QAOA using multiple random initial points for statistical analysis.

        Parameters:
        qaoa (TFG_QAOA): The QAOA instance to optimize.
        num_runs (int): Number of different random initial points.
        initial_point_strategy (str): Strategy for generating initial points ('uniform', 'perturbed', 'gaussian').

        Returns:
        List[QAOARun]: List of QAOARun instances for each run.
        """
        qaoa_runs = []

        params_per_level = qaoa.num_cost_terms + qaoa.num_mixer_terms
        
        for run in range(num_runs):
            if initial_point_strategy == 'uniform':
                initial_point = np.random.uniform(0, 2 * np.pi, size=params_per_level*qaoa.reps)
            elif initial_point_strategy == 'perturbed':
                base_point = np.random.uniform(0, 2 * np.pi, size=params_per_level*qaoa.reps)
                perturbation = np.random.normal(0, 0.1, size=params_per_level*qaoa.reps)
                initial_point = base_point + perturbation
            elif initial_point_strategy == 'gaussian':
                mean_point = np.full(qaoa.reps * params_per_level, np.pi)
                initial_point = np.random.normal(mean_point, 0.1, size=params_per_level*qaoa.reps)
            else:
                raise ValueError("Invalid initial point strategy")

            # Minimize directly:
            start_time = time.time()
            result_optimization = self.run_job(qaoa, qaoa.qaoa_ansatz, initial_point)
            elapsed_time = time.time() - start_time

            # In this case, the vector of elapsed times is 0 on the first p-1 levels:
            elapsed_times = [0] * (qaoa.reps - 1) + [elapsed_time]
            
            qaoa_run = QAOARun(
                initial_params=initial_point,
                qaoa_config=qaoa.qaoa_options,
                model_name="QAOA_Model",
                reps=qaoa.reps,
                num_iterations=-1,
                final_params=result_optimization.x,
                optimizer=self.optimizer_options['name'],
                final_value=result_optimization.fun,
                elasped_time=elapsed_times,
                cost_operator=qaoa.cost_operator,
                mixer_operator=qaoa.mixer_operator,
                initial_state=qaoa.initial_state
            )
            qaoa_runs.append(qaoa_run)
        
        return qaoa_runs





    def optimize_p1(self,
                    qaoa: TFG_QAOA,
                    runs: int = 1,
                    initial_point: np.ndarray = None,
    ):
        best_p1 = None
        best_energy = np.inf

        ansatz_at_p1 = qaoa.build_qaoa_ansatz(1)

        for i in range(runs):
            if initial_point is None:
                initial_point = np.random.rand(qaoa.num_cost_terms + qaoa.num_mixer_terms)
            else:
                # Initial point + some perturbation?:
                initial_point = initial_point #+ 0.1*np.random.rand(2)

            # Measure the time it took to optimize
            start_time = time.time()

            result = self.run_job(qaoa, ansatz_at_p1, initial_point)
            
            elapsed_time = time.time() - start_time


            if result.fun < best_energy:
                best_energy = result.fun
                best_p1 = result.x

        return best_energy, best_p1, elapsed_time


    

    def cost_landscape_p1(
        self,
        qaoa: TFG_QAOA,
        angles={"gamma": np.linspace(0.1, 2*np.pi-0.1, 20), "beta": np.linspace(0.1, 2*np.pi-0.1, 20)},
        withPlot=False
    ):        
        if qaoa.qaoa_options['multiangle']:
            raise ValueError("Cannot use landscape strategy with multiangle option")

        # p = 1 version of our QAOA
        var_form = QAOAAnsatz(qaoa.cost_operator, 1, qaoa.initial_state, qaoa.mixer_operator)

        # Prepare grid for parameters
        p1, p2 = np.meshgrid(angles['gamma'], angles['beta'])
        energies = np.zeros(p1.shape)

        minimal_energy = np.inf
        min_params = None

        # Compute energy for each parameter configuration
        for i in range(len(angles['gamma'])):
            for j in range(len(angles['beta'])):
                energies[i, j] = self.compute_energy(var_form,[p1[i, j], p2[i, j]])
                if energies[i, j] < minimal_energy:
                    minimal_energy = energies[i, j]
                    min_params = [p1[i, j], p2[i, j]]

        # Plotting
        if withPlot:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(p1, p2, energies, cmap='viridis')
            ax.set_xlabel('Parameter 1')
            ax.set_ylabel('Parameter 2')
            ax.set_zlabel('Energy')
            ax.set_title('QAOA Energy Landscape')
            plt.show()

        return minimal_energy, min_params
    
    def sample_best_run(self,
                        best_run: dict,
                        qaoa: TFG_QAOA,
                        group: bool = False,
    ):
        # Sample the best run
        sol_circuit = qaoa.build_qaoa_ansatz(best_run["best_p"]).assign_parameters(best_run["best_params"])
        sol_circuit.measure_all()

        best_samp_dist = self.sampler.run(sol_circuit).result().quasi_dists[0].binary_probabilities()

        if not group:
            return best_samp_dist
        
        # Group the results
        best_samp_dist_grouped = {}
        for key, value in best_samp_dist.items():
            opposite_key = key[::-1]
            key_grouped = key if key < opposite_key else opposite_key
            if key_grouped not in best_samp_dist_grouped:
                best_samp_dist_grouped[key_grouped] = 0
            best_samp_dist_grouped[key_grouped] += value

        return best_samp_dist_grouped
    
    def sample_all_runs(self,
                        qaoa_runs: List[QAOARun],
                        qaoa: TFG_QAOA,
                        group: bool = False,
    ):
        all_samp_dist = []

        for run in qaoa_runs:
            sol_circuit = qaoa.build_qaoa_ansatz(run.num_iterations).assign_parameters(run.final_params)
            sol_circuit.measure_all()

            samp_dist = self.sampler.run(sol_circuit).result().quasi_dists[0].binary_probabilities()

            if not group:
                all_samp_dist.append(samp_dist)
                continue

            # Group the results
            samp_dist_grouped = {}
            for key, value in samp_dist.items():
                opposite_key = key[::-1]
                key_grouped = key if key < opposite_key else opposite_key
                if key_grouped not in samp_dist_grouped:
                    samp_dist_grouped[key_grouped] = 0
                samp_dist_grouped[key_grouped] += value

            all_samp_dist.append(samp_dist_grouped)

        return all_samp_dist
    

    ###########################################################################################################
    ###########################################################################################################
    
        
    def compute_energy(self, qaoa, var_form, params):

        # Use the cost function routine to compute the energy
        return qaoa.cost_function(params, var_form, self.cost_operator, self.estimator)
    

    def _build_sampler(self):
        if self.sampler_options['execution'] == 'local':
            self.sampler = LocalSampler(options={'shots': self.sampler_options['shots']})
        elif self.sampler_options['execution'] == 'runtime':
            raise ValueError("Runtime sampler not implemented")
        else:
            raise ValueError("Invalid execution option")
        return
    

    def _build_estimator(self):
        if self.estimator_options['execution'] == 'local':
            self.estimator = LocalEstimator(options={'shots': self.estimator_options['shots']})
        elif self.estimator_options['execution'] == 'runtime':
            raise ValueError("Runtime estimator not implemented")
        else:
            raise ValueError("Invalid execution option")
        return