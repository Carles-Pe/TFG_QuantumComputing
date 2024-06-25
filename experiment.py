import pickle 
import os
import datetime
from typing import List, Tuple
from datetime import datetime
import numpy as np
import networkx as nx
import json

from TFG_QuantumComputing.models import TFG_QuadraticModel
from TFG_QuantumComputing.qaoa import QAOARun
from TFG_QuantumComputing.test_cases import TFG_ITestCase, TFG_TestCaseSampler
from TFG_QuantumComputing.utils import construct_qaoa_and_solver_from_QAOARun

import matplotlib.pyplot as plt


class Experiment():
    def __init__(self, 
                 name, 
                 description, 
                 data,
                 graph: nx.Graph=None,
                 problem: TFG_QuadraticModel=None):
        self.name = name
        self.dateTime = datetime.now()
        self.description = description
        self.data = data
        self.graph = graph
        self.problem = problem

    def to_dict(self):
        return {
            'name': self.name,
            'dateTime': self.dateTime.isoformat(),
            'description': self.description,
            'data': {key: [qaoa_run.to_dict() for qaoa_run in value] for key, value in self.data.items()} if self.data else {},
            'graph': nx.node_link_data(self.graph) if self.graph else None,
            'problem': self.problem.to_dict() if self.problem else None  # Assuming TFG_QuadraticModel has a to_dict method
        }

    @classmethod
    def from_dict(cls, dict_data):
        experiment = cls(
            name=dict_data['name'],
            description=dict_data['description'],
            data={key: [QAOARun.from_dict(qaoa_data) for qaoa_data in value] for key, value in dict_data['data'].items()},
            graph=nx.node_link_graph(dict_data['graph']) if dict_data['graph'] else None,
            problem=TFG_QuadraticModel.from_dict(dict_data['problem']) if dict_data['problem'] else None  # Assuming TFG_QuadraticModel has a from_dict method
        )
        experiment.dateTime = datetime.fromisoformat(dict_data['dateTime'])
        return experiment



class ExperimentDictionary():
    def __init__(self, 
                 working_directory: str = "experiments",
                 dictionary_path: str = "__dictionary__.pkl"):
        # Stores a Dict[Tuple[str,datetime], file_path]

        self.working_directory = working_directory
        self.dictionary_path = dictionary_path
        self.experiments = {}

        # Working directory must exist, if not, create it:
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)

    def sanitize_filename(self, s):
        if os.name == 'nt':
            return s.replace(':','_').replace(' ','_')
        return s

    def load_dictionary(self):
        # Read from the file called working_directory/dictionary_path
        try:
            with open(os.path.join(self.working_directory, self.dictionary_path), 'rb') as f:
                self.experiments = pickle.load(f)
        except FileNotFoundError:
            # If the file is not found, create a new one
            with open(os.path.join(self.working_directory, self.dictionary_path), 'wb') as f:
                pickle.dump(self.experiments, f)

        

    def add_experiment(self, experiment: Experiment):
        # Save Experiment in a file with the name_datetime of the experiment:
        file_path = f"{experiment.name}_{experiment.dateTime}.pkl"
        with open(os.path.join(self.working_directory, file_path), 'wb') as f:
            pickle.dump(experiment, f)

        # Add the experiment to the dictionary, dateTime to string
        self.experiments[(experiment.name, str(experiment.dateTime))] = file_path

        # Save the dictionary in a file called dictionary_path
        with open(os.path.join(self.working_directory, self.dictionary_path), 'wb') as f:
            pickle.dump(self.experiments, f)

    
    def get_experiment(self, name, dateTime):

        # Check if the key exists:
        if (name, dateTime) not in self.experiments:
            raise KeyError(f"No experiment with name {name} and dateTime {dateTime} found")

        # Get the file_path from the dictionary
        file_path = self.experiments[(name, str(dateTime))]

        # Load the data from the file, if not found, raise error:
        try:
            sanitized_filename = self.sanitize_filename(file_path)
            print("Attemoting to load experiment from file:", sanitized_filename)
            with open(os.path.join(self.working_directory, sanitized_filename), 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No {file_path} file found in directory {self.working_directory}, integrity compromised")

        # Return the data
        return data
    
    def get_experiment_by_tuple(self, key: Tuple[str, datetime]):
        if isinstance(key, Tuple) and len(key) == 2 and isinstance(key[0], str) and (isinstance(key[1], datetime) or isinstance(key[1], str)):
            return self.get_experiment(key[0], key[1])
        else:
            raise ValueError("Key must be a tuple of (str, datetime) or (str, str)")

    
    
    def backup_to_json(self):
        for (name, dateTime), file_path in self.experiments.items():
            experiment = self.get_experiment(name, dateTime)
            json_file_path = file_path.replace('.pkl', '.json')
            try:
                with open(os.path.join(self.working_directory, json_file_path), 'w') as f:
                    json.dump(experiment.to_dict(), f)
            except TypeError as e:
                print(f"Failed to serialize experiment: {name} at {dateTime}")
                print(f"Problematic data: {experiment.to_dict()}")
                raise e  # Re-raise the exception after logging the data

        json_dict_path = self.dictionary_path.replace('.pkl', '.json')
        try:
            with open(os.path.join(self.working_directory, json_dict_path), 'w') as f:
                json.dump({f"{name}_{dateTime}": file_path for (name, dateTime), file_path in self.experiments.items()}, f)
        except TypeError as e:
            print(f"Failed to serialize experiment dictionary")
            print(f"Problematic dictionary: {self.experiments}")
            raise e  # Re-raise the exception after logging the data
        
        
    def load_from_json(self):
        json_files = [f for f in os.listdir(self.working_directory) if f.endswith('.json') and f != self.dictionary_path.replace('.pkl', '.json')]
        for json_file in json_files:
            with open(os.path.join(self.working_directory, json_file), 'r') as f:
                data = json.load(f)
                experiment = Experiment.from_dict(data)
                self.add_experiment(experiment)

        # Loading dictionary
        json_dict_path = self.dictionary_path.replace('.pkl', '.json')
        try:
            with open(os.path.join(self.working_directory, json_dict_path), 'r') as f:
                self.experiments = {(name, dateTime): file_path for (name, dateTime), file_path in json.load(f).items()}
        except FileNotFoundError:
            raise FileNotFoundError(f"No {json_dict_path} file found")

    def get_experiment_names(self):
        return list(self.experiments.keys())
    

    def get_experiment_files(self):
        return list(self.experiments.values())
    
    
    def get_dictionary(self):
        return self.experiments
    

    def merge_with_dictionary(self, second_dictionary_path: str):
        # Load the second dictionary
        try:
            with open(os.path.join(self.working_directory, second_dictionary_path), 'rb') as f:
                second_dictionary = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No {second_dictionary_path} file found")
        
        # Merge the two dictionaries
        self.experiments.update(second_dictionary)

        # Save the dictionary in a file called dictionary_path
        with open(os.path.join(self.working_directory, self.dictionary_path), 'wb') as f:
            pickle.dump(self.experiments, f)

        # Delete the second dictionary file
        os.remove(second_dictionary_path)

def plot_experiment_uniform_perturbed_gaussian_maxreps_only(exp: Experiment, 
                                                            anlytical_threshold: float = None, 
                                                            store_in_folder: str = None,
                                                            max_rep_filter: bool = True):

    qaoa_runs_uniform = exp.data['uniform']
    qaoa_runs_perturbed = exp.data['perturbed']
    qaoa_runs_gaussian = exp.data['gaussian']

    # Get the max run.reps for all runs:
    max_reps = max([run.reps for run in qaoa_runs_uniform])

    # Filter the runs with the max reps
    if max_rep_filter:
        qaoa_runs_uniform = [run for run in qaoa_runs_uniform if run.reps == max_reps]
        qaoa_runs_perturbed = [run for run in qaoa_runs_perturbed if run.reps == max_reps]
        qaoa_runs_gaussian = [run for run in qaoa_runs_gaussian if run.reps == max_reps]


    qaoa, solver = construct_qaoa_and_solver_from_QAOARun(qaoa_runs_uniform[0])



    best_run_uniform = min(qaoa_runs_uniform, key=lambda x: x.final_value)
    best_run_perturbed = min(qaoa_runs_perturbed, key=lambda x: x.final_value)
    best_run_gaussian = min(qaoa_runs_gaussian, key=lambda x: x.final_value)

    # Sample the best run
    best_run_data = {
        "best_p": best_run_uniform.reps,
        "best_params": best_run_uniform.final_params,
        "best_energy": best_run_uniform.final_value
    }
    best_samp_dist_uniform = solver.sample_best_run(best_run_data, qaoa, group=True)

    best_run_data = {
        "best_p": best_run_perturbed.reps,
        "best_params": best_run_perturbed.final_params,
        "best_energy": best_run_perturbed.final_value
    }
    best_samp_dist_perturbed = solver.sample_best_run(best_run_data, qaoa, group=True)

    best_run_data = {
        "best_p": best_run_gaussian.reps,
        "best_params": best_run_gaussian.final_params,
        "best_energy": best_run_gaussian.final_value
    }
    best_samp_dist_gaussian = solver.sample_best_run(best_run_data, qaoa, group=True)

    # Plot the results, with the same y-axis in each subplot
    # Add labels to the x-axis and y-axis
    fig, axs = plt.subplots(1, 3, figsize=(30, 20), sharey=True)
    axs[0].bar(best_samp_dist_uniform.keys(), best_samp_dist_uniform.values())
    axs[0].set_title('Uniform Best: ' + str(np.round(best_run_uniform.final_value, decimals=3)))
    axs[0].xaxis.set_tick_params(rotation=45)

    axs[1].bar(best_samp_dist_perturbed.keys(), best_samp_dist_perturbed.values())
    axs[1].set_title('Perturbed Best: ' + str(np.round(best_run_perturbed.final_value, decimals=3)))
    axs[1].xaxis.set_tick_params(rotation=45)
    

    axs[2].bar(best_samp_dist_gaussian.keys(), best_samp_dist_gaussian.values())
    axs[2].set_title('Gaussian Best: ' + str(np.round(best_run_gaussian.final_value, decimals=3)))
    axs[2].xaxis.set_tick_params(rotation=45)
    

    #fig.suptitle(exp.name+(' Max Reps Filtered' if max_rep_filter else ''), fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if store_in_folder is not None:
        plt.savefig(f"{store_in_folder}/{exp.name}_best_runs{'_maxRepFiltered'if max_rep_filter else ''}.png")
    plt.show()
    plt.close()

    _,offset = exp.problem.translate_qp_to_ising()

    if anlytical_threshold is not None:
        
        fun_values_uniform = [(run.final_value + offset)/(anlytical_threshold) for run in qaoa_runs_uniform]
        fun_values_perturbed = [(run.final_value + offset)/(anlytical_threshold) for run in qaoa_runs_perturbed]
        fun_values_gaussian = [(run.final_value + offset)/(anlytical_threshold) for run in qaoa_runs_gaussian]

    else:
        fun_values_uniform = [run.final_value + offset for run in qaoa_runs_uniform]
        fun_values_perturbed = [run.final_value + offset for run in qaoa_runs_perturbed]
        fun_values_gaussian = [run.final_value + offset for run in qaoa_runs_gaussian]

    # Take into account it is a continuous variable
    fig, axs = plt.subplots(1, 3, figsize=(15, 10), sharey=True)
    axs[0].hist(fun_values_uniform, bins=20)
    axs[0].set_title('Uniform Best: ' + str(np.round(best_run_uniform.final_value + offset, decimals=3)))
    axs[0].set_ylabel('Frequency')
    axs[0].set_xlabel('Energy ranges')
    
    axs[1].hist(fun_values_perturbed, bins=20)
    axs[1].set_title('Perturbed Best: ' + str(np.round(best_run_perturbed.final_value + offset, decimals=3)))
    axs[1].set_xlabel('Energy ranges')

    axs[2].hist(fun_values_gaussian, bins=20)
    axs[2].set_title('Gaussian Best: ' + str(np.round(best_run_gaussian.final_value + offset, decimals=3)))
    axs[2].set_xlabel('Energy ranges')

    fig.suptitle(exp.name+(' Max Reps Filtered' if max_rep_filter else ''), fontsize=16)
    if store_in_folder is not None:
        plt.savefig(f"{store_in_folder}/{exp.name}_fun_values{'_maxRepFiltered'if max_rep_filter else ''}.png")
    plt.show()
    plt.close()


def plot_hist_as_line(fig,ax,data, label, color, rescaleFactor=None):
        counts, bin_edges = np.histogram(data, bins='auto')
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        if rescaleFactor is not None:
            counts = counts * rescaleFactor
        ax.plot(bin_centers, counts, label=label, color=color)



def plot_experiment_comparison_uniform_perturbed_gaussian_maxreps_only(exps: List[Experiment], 
                                                                       anlytical_threshold: float = None, 
                                                                       store_in_folder: str = None,
                                                                       max_rep_filter: bool = True):

    colors = [
    'blue',
    'red',
    'green',
    'purple'
    ]
    # Take into account it is a continuous variable
    fig, axs = plt.subplots(1, 3, figsize=(15, 10), sharey=True)


    for idx, exp in enumerate(exps):

        qaoa_runs_uniform = exp.data['uniform']
        qaoa_runs_perturbed = exp.data['perturbed']
        qaoa_runs_gaussian = exp.data['gaussian']

        # Get the max run.reps for all runs:
        max_reps = max([run.reps for run in qaoa_runs_uniform])

        # If max_rep_filter is True, we must rescale the histogram because the optimize methods will have many more results and plots so that they are all nomralized
        # Max reps = p of the experiment. The we know that if a Straight Minimization has done 10 runs, an Optimize method has done (p-1)*10 runs
        rescale_factor = 1/(max_reps-1) if exp.name.startswith('Optimize') and not max_rep_filter else 1


        if max_rep_filter:
            # Filter the runs with the max reps
            qaoa_runs_uniform = [run for run in qaoa_runs_uniform if run.reps == max_reps]
            qaoa_runs_perturbed = [run for run in qaoa_runs_perturbed if run.reps == max_reps]
            qaoa_runs_gaussian = [run for run in qaoa_runs_gaussian if run.reps == max_reps]


        qaoa, solver = construct_qaoa_and_solver_from_QAOARun(qaoa_runs_uniform[0])



        best_run_uniform = min(qaoa_runs_uniform, key=lambda x: x.final_value)
        best_run_perturbed = min(qaoa_runs_perturbed, key=lambda x: x.final_value)
        best_run_gaussian = min(qaoa_runs_gaussian, key=lambda x: x.final_value)


        _,offset = exp.problem.translate_qp_to_ising()

        if anlytical_threshold is not None:
            
            fun_values_uniform = [(run.final_value + offset)/(anlytical_threshold) for run in qaoa_runs_uniform]
            fun_values_perturbed = [(run.final_value + offset)/(anlytical_threshold) for run in qaoa_runs_perturbed]
            fun_values_gaussian = [(run.final_value + offset)/(anlytical_threshold) for run in qaoa_runs_gaussian]

        else:
            fun_values_uniform = [run.final_value + offset for run in qaoa_runs_uniform]
            fun_values_perturbed = [run.final_value + offset for run in qaoa_runs_perturbed]
            fun_values_gaussian = [run.final_value + offset for run in qaoa_runs_gaussian]

        
        plot_hist_as_line(fig,axs[0],fun_values_uniform, exp.name, colors[idx], rescaleFactor= rescale_factor)
        axs[0].set_title('Uniform fun values, comaprison')
        axs[0].set_ylabel('Frequency')
        axs[0].set_xlabel('Energy ranges')

        plot_hist_as_line(fig,axs[1],fun_values_perturbed, exp.name, colors[idx], rescaleFactor= rescale_factor)
        axs[1].set_title('Perturbed fun values, comaprison')
        axs[1].set_xlabel('Energy ranges')

        plot_hist_as_line(fig,axs[2],fun_values_gaussian, exp.name, colors[idx], rescaleFactor= rescale_factor)
        axs[2].set_title('Gaussian fun values, comaprison')
        axs[2].set_xlabel('Energy ranges')

    fig.suptitle('4 Method comparison'+(' Max Reps Filtered' if max_rep_filter else ''), fontsize=16)
    # Legend:
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    if store_in_folder is not None:
        plt.savefig(f"{store_in_folder}/comparison_fun_values{'_maxRepFiltered' if max_rep_filter else ''}.png")
    plt.show()
    plt.close()





def plot_experiment_map_uniform_perturbed_gaussian_maxreps_only(exp: Experiment, 
                                                                examples: TFG_TestCaseSampler, 
                                                                store_in_folder: str = None,
                                                                max_rep_filter: bool = True):

    qaoa_runs_uniform = exp.data['uniform']
    qaoa_runs_perturbed = exp.data['perturbed']
    qaoa_runs_gaussian = exp.data['gaussian']

    # Get the max run.reps for all runs:
    max_reps = max([run.reps for run in qaoa_runs_uniform])

    # Filter the runs with the max reps
    if max_rep_filter:
        qaoa_runs_uniform = [run for run in qaoa_runs_uniform if run.reps == max_reps]
        qaoa_runs_perturbed = [run for run in qaoa_runs_perturbed if run.reps == max_reps]
        qaoa_runs_gaussian = [run for run in qaoa_runs_gaussian if run.reps == max_reps]


    qaoa, solver = construct_qaoa_and_solver_from_QAOARun(qaoa_runs_uniform[0])



    best_run_uniform = min(qaoa_runs_uniform, key=lambda x: x.final_value)
    best_run_perturbed = min(qaoa_runs_perturbed, key=lambda x: x.final_value)
    best_run_gaussian = min(qaoa_runs_gaussian, key=lambda x: x.final_value)

    # Sample the best run
    best_run_data = {
        "best_p": best_run_uniform.reps,
        "best_params": best_run_uniform.final_params,
        "best_energy": best_run_uniform.final_value
    }
    best_samp_dist_uniform = solver.sample_best_run(best_run_data, qaoa, group=True)
    # Extract the most probable solution
    best_solution_uniform = max(best_samp_dist_uniform, key=best_samp_dist_uniform.get)
    # COnvert to list of ints
    best_solution_uniform = [int(node) for node in best_solution_uniform]

    best_run_data = {
        "best_p": best_run_perturbed.reps,
        "best_params": best_run_perturbed.final_params,
        "best_energy": best_run_perturbed.final_value
    }
    best_samp_dist_perturbed = solver.sample_best_run(best_run_data, qaoa, group=True)
    # Extract the most probable solution
    best_solution_perturbed = max(best_samp_dist_perturbed, key=best_samp_dist_perturbed.get)
    # COnvert to list of ints
    best_solution_perturbed = [int(node) for node in best_solution_perturbed]

    best_run_data = {
        "best_p": best_run_gaussian.reps,
        "best_params": best_run_gaussian.final_params,
        "best_energy": best_run_gaussian.final_value
    }
    best_samp_dist_gaussian = solver.sample_best_run(best_run_data, qaoa, group=True)
    # Extract the most probable solution
    best_solution_gaussian = max(best_samp_dist_gaussian, key=best_samp_dist_gaussian.get)
    # COnvert to list of ints
    best_solution_gaussian = [int(node) for node in best_solution_gaussian]

    # Plot the results, with the same y-axis in each subplot
    # Print only for the best strategy, compare the 3 best_run_data

    idx = np.argmin([best_run_uniform.final_value, best_run_perturbed.final_value, best_run_gaussian.final_value])

    solution_to_plot = [best_solution_uniform, best_solution_perturbed, best_solution_gaussian][idx]
    strategy_to_plot = ['uniform', 'perturbed', 'gaussian'][idx]


    print(f"Best solution found with strategy {strategy_to_plot}{' Max Reps Filtered' if max_rep_filter else ''}: {solution_to_plot}")
    examples.plot_case(exp.graph, solution_to_plot, store_in_folder=store_in_folder)

    
