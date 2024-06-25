#%% Set Up for experiments

import uuid
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

n = 11
if n > 11:
    raise ValueError("The number of nodes is too high for the brute force method")
p_value = 15
number_of_graphs = 100
tasks_per_thread = 25
perform_warmstart = False

print(f"Starting the experiment with p={p_value}, {number_of_graphs} different graphs, {tasks_per_thread} tasks per thread, warm start: {perform_warmstart}, on a number of nodes: {n}")


graphs = [TFG_TestCaseSampler("scandinavia").get_test_case().select_case(f"scandinavian_{n}") for _ in range(number_of_graphs)]


# %%

# problem = TFG_SimplePowerModel_2Partitions(G)
# qubitOp, offset = problem.translate_qp_to_ising()


# %%

solver = TFG_Solver(
                    optimizer_options={
                        'name': 'COBYLA',
                        'disp': False
                        })

manager = Manager()
exp_list = manager.list()  # Create a managed list

exp_dict = ExperimentDictionary(working_directory=f'4Methods_n{n}_p{p_value}_{number_of_graphs}graphs{"_WarmStart" if perform_warmstart else ""}')

id_graph_warmStart = []

# %% Add warm start strategy, only not multiangle
for G in graphs:
    problem = TFG_SimplePowerModel_2Partitions(G)
    warm_start_values = solver.CPLEX_solver(problem) if perform_warmstart else None
    id_graph_warmStart.append((uuid.uuid4(),(G, warm_start_values)))


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
tasks = [(params, strategy, graph_warm, graph_id) for graph_id, graph_warm in id_graph_warmStart for params in experiment_params for strategy in strategies]

def run_experiment(params, strategy, graph_warm, graph_id):
    G, warm_start_values = graph_warm
    params['qaoa_options']['warm_start'] = warm_start_values
    problem = TFG_SimplePowerModel_2Partitions(G)
    qaoa = TFG_QAOA(cost_operator=problem.qubitOp, reps=p_value, qaoa_options=params['qaoa_options'])
    
    print(f"Running optimization with {strategy} initial points for {params['name']}")
    if params['function'] == 'straight_minimization':
        qaoa_runs = solver.straight_minimization(qaoa, num_runs=1, initial_point_strategy=strategy)
    else:
        qaoa_runs = solver.optimize_with_only_start_num_runs(qaoa, num_runs=1, initial_point_strategy=strategy)

    return (params, strategy, qaoa_runs, G, graph_id)


# %%

results = []
with ProcessPoolExecutor(max_workers=12*(number_of_graphs // tasks_per_thread)) as executor:
    futures = [executor.submit(run_experiment, params, strategy, graph_warm, graph_id) for params, strategy, graph_warm, graph_id in tasks]
    for future in as_completed(futures):
        results.append(future.result())  # collect results



# %%

# Group results by experiment name
grouped_results = {}
for params, strategy, qaoa_runs, G, graph_id in results:
    key_name = params['name']+f'graph_id={graph_id}'
    if key_name not in grouped_results:
        grouped_results[key_name] = {
            'description': params['description'],
            'data': {},
            'graph': G,
            'problem': problem
        }
    # If key not exists
    if strategy not in grouped_results[key_name].keys():
        grouped_results[key_name]['data'][strategy] = qaoa_runs
    else:
        grouped_results[key_name]['data'][strategy].extend(qaoa_runs)


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

# %%

# best_run_uniform = min(qaoa_runs_uniform, key=lambda x: x.final_value)
# best_run_perturbed = min(qaoa_runs_perturbed, key=lambda x: x.final_value)
# best_run_gaussian = min(qaoa_runs_gaussian, key=lambda x: x.final_value)

# # Sample the best run
# best_run_data = {
#     "best_p": best_run_uniform.reps,
#     "best_params": best_run_uniform.final_params,
#     "best_energy": best_run_uniform.final_value
# }
# best_samp_dist_uniform = solver.sample_best_run(best_run_data, qaoa, group=True)

# best_run_data = {
#     "best_p": best_run_perturbed.reps,
#     "best_params": best_run_perturbed.final_params,
#     "best_energy": best_run_perturbed.final_value
# }
# best_samp_dist_perturbed = solver.sample_best_run(best_run_data, qaoa, group=True)

# best_run_data = {
#     "best_p": best_run_gaussian.reps,
#     "best_params": best_run_gaussian.final_params,
#     "best_energy": best_run_gaussian.final_value
# }
# best_samp_dist_gaussian = solver.sample_best_run(best_run_data, qaoa, group=True)

# fun_values_uniform = [run.final_value + offset for run in qaoa_runs_uniform]
# fun_values_perturbed = [run.final_value + offset for run in qaoa_runs_uniform]
# fun_values_gaussian = [run.final_value + offset for run in qaoa_runs_uniform]

# # Take into account it is a continuous variable
# fig, axs = plt.subplots(1, 3, figsize=(15, 10), sharey=True)
# axs[0].hist(fun_values_uniform, bins=20)
# axs[0].set_title('Uniform fun values, best: ' + str(best_run_uniform.fin
