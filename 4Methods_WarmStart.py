#%% Set Up for experiments

from TFG_QuantumComputing.test_cases import TFG_TestCaseSampler
from TFG_QuantumComputing.models import TFG_SimplePowerModel_2Partitions
from TFG_QuantumComputing.brute_force import TFG_BruteForce 
from TFG_QuantumComputing.qaoa import TFG_QAOA
from TFG_QuantumComputing.solver import TFG_Solver
from TFG_QuantumComputing.experiment import Experiment, ExperimentDictionary

import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

# %%

test_mode = True
p_value = 3
number_of_runs = 1
runs_per_thread = 1
perform_warmstart = False

print(f"Starting the experiment with p={p_value}, {number_of_runs} runs, {runs_per_thread} runs per thread, warm start: {perform_warmstart}")


examples = TFG_TestCaseSampler("scandinavia").get_test_case()
G = examples.select_case("scandinavian_8")
examples.plot_case(G)

problem = TFG_SimplePowerModel_2Partitions(G)
qubitOp, offset = problem.translate_qp_to_ising()

# First we see the problem bruteforced, if we don't like, we abort:
brute_force = TFG_BruteForce()
brute_force.solve(problem, generateAll=True)
#brute_force.plot_histogram_solutions(threshold=brute_force.best_cost_brute)

# %%

solver = TFG_Solver(
                    optimizer_options={
                        'name': 'COBYLA',
                        'disp': False
                        })

manager = Manager()
exp_list = manager.list()  # Create a managed list

exp_dict = ExperimentDictionary(working_directory=f'{'Test_' if test_mode else ''}4Methods_p{p_value}_{number_of_runs}runs{"_WarmStart" if perform_warmstart else ""}')


# %% Add warm start strategy, only not multiangle
warm_start_values = solver.CPLEX_solver(problem) if perform_warmstart else None
print("Warm Start is: ", warm_start_values)

# %%
experiment_params = [
    {
        'name': f'StraightMinimization_Multiangle_p={p_value}',
        'description': f'Straight minimization with multiangle ansatz, p={p_value}',
        'qaoa_options': {'parametrization': None, 'warm_start': warm_start_values, 'multiangle': True},
        'function': 'straight_minimization'
    },
    {
        'name': f'StraightMinimization_p={p_value}',
        'description': f'Straight minimization with normal ansatz, p={p_value}',
        'qaoa_options': {'parametrization': None, 'warm_start': warm_start_values, 'multiangle': False},
        'function': 'straight_minimization'
    },
    {
        'name': f'Optimize_p={p_value}',
        'description': f'Optimize with normal ansatz, p={p_value}',
        'qaoa_options': {'parametrization': None, 'warm_start': warm_start_values, 'multiangle': False},
        'function': 'optimize_with_only_start_num_runs'
    },
    {
        'name': f'Optimize_Fourier_p={p_value}',
        'description': f'Optimize Fourier parametrization with normal ansatz, p={p_value}',
        'qaoa_options': {'parametrization': 'fourier', 'warm_start': warm_start_values, 'multiangle': False},
        'function': 'optimize_with_only_start_num_runs'
    }
]

strategies = ['uniform', 'perturbed', 'gaussian']


# %% 
# Combine parameters and strategies
tasks = [(params, strategy) for _ in range(number_of_runs) for params in experiment_params for strategy in strategies]

def run_experiment(params, strategy):
    qaoa = TFG_QAOA(cost_operator=problem.qubitOp, reps=p_value, qaoa_options=params['qaoa_options'])
    
    print(f"Running optimization with {strategy} initial points for {params['name']}")
    if params['function'] == 'straight_minimization':
        qaoa_runs = solver.straight_minimization(qaoa, num_runs=1, initial_point_strategy=strategy)
    else:
        qaoa_runs = solver.optimize_with_only_start_num_runs(qaoa, num_runs=1, initial_point_strategy=strategy)

    return (params, strategy, qaoa_runs)


# %%

results = []
with ProcessPoolExecutor(max_workers=12*(number_of_runs // runs_per_thread)) as executor:
    futures = [executor.submit(run_experiment, params, strategy) for params, strategy in tasks]
    for future in as_completed(futures):
        results.append(future.result())  # collect results



# %%

# Group results by experiment name
grouped_results = {}
for params, strategy, qaoa_runs in results:
    if params['name'] not in grouped_results:
        grouped_results[params['name']] = {
            'description': params['description'],
            'data': {},
            'graph': G,
            'problem': problem
        }
    # If key not exists
    if strategy not in grouped_results[params['name']]['data'].keys():
        grouped_results[params['name']]['data'][strategy] = qaoa_runs
    else:
        grouped_results[params['name']]['data'][strategy].extend(qaoa_runs)


# %%

# Create experiments from grouped results
for name, details in grouped_results.items():
    exp = Experiment(
        name=name,
        description=details['description'],
        data=details['data'],
        graph=details['graph'],
        problem=details['problem']
    )
    exp_list.append(exp)

# %%

# Add the experiments from the shared list to the ExperimentDictionary
for exp in exp_list:
    exp_dict.add_experiment(exp)

# %%
# print the dictionary
print(exp_dict.experiments)

