
from .models import TFG_QuadraticModel

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from qiskit_optimization.problems.quadratic_program import QuadraticProgram

class TFG_BruteForce():
    def __init__(self) -> None:
        self.brute_force_solve = None
        self.xbest_brute = None
        self.best_cost_brute = None
        return
    
    def solve(self,
              problem: TFG_QuadraticModel,
              useTheoretical: bool = False,
              generateAll: bool = False) -> None:
        """
        Solve the MaxCut problem using brute force.

        Parameters:
        problem (TFG_QuadraticModel): The quadratic model of the problem.
        useTheoretical (bool): Flag to use the theoretical quadratic model.
        generateAll (bool): Flag to generate all possible solutions.
        """
        
        n = problem.nqubits

        qp = problem.quadratic_model_theoretical if useTheoretical else problem.quadratic_model


        if generateAll:
            self.brute_force_solve = []


        xbest = None
        best_cost = np.inf
        feasible_solutions = 0

        for b in range(2**n):
            # Convert to binary, remove the '0b' prefix, zero fill till n bits, and reverse the list
            # So that bit at index 0 is the least significant bit
            x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
            cost, feasible = self.evaluate_solution(qp, x)

            
            if feasible:
                if generateAll:
                    self.brute_force_solve.append((x, cost))
                feasible_solutions += 1
                if cost < best_cost:
                    xbest = x
                    best_cost = cost

        self.xbest_brute = xbest
        self.best_cost_brute = best_cost
        if generateAll:
            print("[*] Ordering Solutions...(this may take a while)")
            self.brute_force_solve.sort(key=lambda x: x[1], reverse=False)
            print("[*] Solutions ordered")
        return


    def evaluate_solution(self,
                          qp: QuadraticProgram, 
                          solution: List[int]) -> Tuple[float, bool]:
        """
        Evaluate a given binary solution in the context of a QuadraticProgram.

        Parameters:
        qp (QuadraticProgram): The QuadraticProgram object.
        solution (list): The solution to evaluate, given as a list of binary values.

        Returns:
        Tuple[float, bool]: The objective value for the given solution and its feasibility.
        """
        # Convert the solution list to a dictionary format expected by OptimizationResult
        var_names = [var.name for var in qp.variables]
        solution_dict = dict(zip(var_names, solution))
        
        # Evaluate the objective value using the solution
        objective_value = qp.objective.evaluate(solution_dict)
        
        # Optionally, you can also check if the solution is feasible
        is_feasible = qp.is_feasible(solution)
        
        return objective_value, is_feasible
    
    def balance_deviation(self,
                          problem: TFG_QuadraticModel, 
                          solution: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the deviation in balance for each partition in the solution.

        Parameters:
        problem (TFG_QuadraticModel): The quadratic model of the problem.
        solution (list): The solution to evaluate, given as a list of binary values.

        Returns:
        Tuple[np.ndarray, np.ndarray]: The partition deviation and surplus deviation.
        """

        expected_k = problem.k

        # Compute the surplus balance for each partition
        if problem.partitions < 2:
            raise ValueError("Partition number must be at least 2")

        surplus_balance = np.zeros(problem.partitions)
        partition_balance = np.zeros(problem.partitions)

        if problem.partitions == 2:
            # If only 2 partitions, problem is easily encoded in 1 bin var per node
            for i, x in enumerate(solution):
                surplus_balance[int(x)] += problem.graph.nodes[i]["weight"]
                partition_balance[int(x)] += 1

        else:
            # The solution is one hot encoded, so we need to decode it
            for i, x in enumerate(solution):
                surplus_balance[i % problem.partitions] += problem.graph.nodes[i]["weight"]
                partition_balance[i % problem.partitions] += 1

        
        # We need the mean:
        for i in range(problem.partitions):
            if partition_balance[i] == 0:
                surplus_balance[i] = -1
            else:
                surplus_balance[i] = surplus_balance[i] / partition_balance[i]

        surplus_deviation = np.zeros(problem.partitions)
        partition_deviation = np.zeros(problem.partitions)

        # Compute the deviation from the expected surplus balance
        for i in range(problem.partitions):
            surplus_deviation[i] = abs(surplus_balance[i] - expected_k)
            partition_deviation[i] = abs(partition_balance[i] - problem.graph.number_of_nodes()/problem.partitions)

        return partition_deviation, surplus_deviation
    
    def plot_histogram_solutions(self,
                                 threshold: float = None,
                                 store_in_folder: str = None) -> None:
        
        """
        Plot a histogram of solution values.

        Parameters:
        threshold (float): Optional threshold value to highlight in the histogram.
        """
        
        if self.brute_force_solve is None:
            raise ValueError("No solutions to plot, execute with generateAll=True")

        # Extract y values from each tuple
        y_values = [y for x, y in self.brute_force_solve]
        
        # Plot histogram of y values
        plt.hist(y_values, bins='auto', alpha=0.7, rwidth=0.85)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Y Values')
        plt.grid(axis='y', alpha=0.75)

        # Plot a vertical line at the threshold value
        if threshold is not None:
            plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
            plt.legend()

        # Show the plot
        if store_in_folder is not None:
            plt.savefig(store_in_folder + '/BruteForceHistogram.png')
        plt.show()
        plt.close()

# %% [markdown]
# #### Brute Force Test

# %%
# Generate a fc_4 graph and solve it with the brute force method
# examples1 = TFG_TestCaseSampler().get_test_case()
# G = examples1.select_case("fc_8")
# problem = TFG_SimplePowerModel_2Partitions(G)
# brute_force = TFG_BruteForce()

# brute_force.solve(problem, useTheoretical=False, generateAll=True)
# brute_force.plot_histogram_solutions(threshold=brute_force.best_cost_brute)
# print(f"Best cost: {brute_force.best_cost_brute} for solution {brute_force.xbest_brute}")
# examples1.plot_case(G, brute_force.xbest_brute)
