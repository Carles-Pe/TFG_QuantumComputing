import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Tuple

from qiskit.quantum_info import Pauli, SparsePauliOp

from docplex.mp.model import Model

from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo


class TFG_QuadraticModel():
    def __init__(self) -> None:
        self.nqubits = -1
        self.k = -1
        self.partitions = -1
        return


    def generate_models(self,
                            alpha: float = 1.0,
                            beta: float = 1.0,
                            k: float = None,
                            lambda_one_hot: float = 1.0,
                            lambda_surplus: float = 1.0) -> None:

            # Generate the quadratic model defining the x_i problem with xi in {0,1}
            self.generate_quadratic_model(alpha, beta, k, lambda_one_hot, lambda_surplus)

            # Generate the quadratic model defining the x_i problem with xi in {0,1}
            # But implementing the constraints per se, so they can be evaluated
            self.generate_quadratic_model_theoretical(alpha, beta, k, lambda_one_hot, lambda_surplus)

            # Generate the CPLEX model, which is used for warmstarting the QAOA
            self.generate_CPLEX_valid_model(alpha, beta, k, lambda_one_hot, lambda_surplus)

            # Generate the Ising model, if the QuadraticProgram -> SparsePauliOp default conversion
            # wnats to be skipped
            self.generate_ising_model(alpha, beta, k, lambda_one_hot, lambda_surplus)


    def generate_quadratic_model(self,
                                alpha: float = 1.0,
                                beta: float = 1.0,
                                k: float = None,
                                lambda_one_hot: float = 1.0,
                                lambda_surplus: float = 1.0) -> None:
        raise ValueError("generate_quadratic_model() method not implemented")


    def generate_quadratic_model_theoretical(self,
                                            alpha: float = 1.0,
                                            beta: float = 1.0,
                                            k: float = None,
                                            lambda_one_hot: float = 1.0,
                                            lambda_surplus: float = 1.0) -> None:
        raise ValueError("generate_quadratic_model_theoretical() method not implemented")


    def generate_CPLEX_valid_model(self,
                                    alpha: float = 1.0,
                                    beta: float = 1.0,
                                    k: float = None,
                                    lambda_one_hot: float = 1.0,
                                    lambda_surplus: float = 1.0) -> None:
        raise ValueError("generate_CPLEX_valid_model() method not implemented")


    def generate_ising_model(self,
                            alpha: float = 1.0,
                            beta: float = 1.0,
                            k: float = None,
                            lambda_one_hot: float = 1.0,
                            lambda_surplus: float = 1.0) -> None:
        raise ValueError("generate_ising_model() method not implemented")


    def translate_qp_to_ising(self) -> None:
        raise ValueError("translate_qp_to_ising() method not implemented")


    def to_dict(self) -> dict:
        raise ValueError("to_dict() method not implemented")
    
    def from_dict(self, data: dict) -> None:
        raise ValueError("from_dict() method not implemented")

class TFG_MaxCut(TFG_QuadraticModel):

    def __init__(self, graph: nx.Graph):
        """
        Initialize the TFG_MaxCut class with a given graph.
        
        Parameters:
        graph (nx.Graph): The input graph for the MaxCut problem.
        """
        super().__init__()

        self.graph = graph
        self.quadratic_model = None
        self.quadratic_model_theoretical = None
        self.CPLEX_model = None
        self.qubitOp = None
        self.offset = None

        self.nqubits = graph.number_of_nodes()
        self.k = np.mean([self.graph.nodes[i]["weight"] for i in self.graph.nodes])
        self.partitions = 2

        self.generate_models()

        

    def generate_models(self,
                        alpha: float = None,
                        beta: float = None,
                        k: float = None,
                        lambda_one_hot: float = None,
                        lambda_surplus: float = None,
                        withSurplusConstraint: bool = False):
        """
        Generate all models required for the MaxCut problem.
        """
        self.generate_quadratic_model()
        self.generate_quadratic_model_theoretical()
        self.generate_CPLEX_model()
        self.generate_ising_model()

    def generate_ising_model(self):
        """
        Generate the Ising model from the given graph.
        """
        pauli_list = []
        for i, j in self.graph.edges:
            z = np.zeros(self.graph.number_of_nodes(), dtype=bool)
            x = np.zeros_like(z, dtype=bool)

            z[i] = True
            z[j] = True

            pauli = Pauli((z, x))
            pauli_list.append((pauli.to_label(), 1.0))

        self.qubitOp = SparsePauliOp.from_list(pauli_list)
        self.qubo = self.qubitOp
        self.offset = 0

    def generate_quadratic_model(self):
        """
        Generate the quadratic model for the MaxCut problem.
        """
        mdl = Model('TFG_MaxCut::QuadraticModel')
        x = {i: mdl.binary_var(name=f'x_{i}') for i in self.graph.nodes}

        objective = mdl.sum(-x[i] * (1 - x[j]) for i, j in self.graph.edges)
        mdl.minimize(objective)

        self.quadratic_model = from_docplex_mp(mdl)

    def generate_quadratic_model_theoretical(self):
        """
        Generate the theoretical quadratic model for the MaxCut problem.
        """
        mdl = Model('TFG_MaxCut::QuadraticModelTheoretical')
        x = {i: mdl.binary_var(name=f'x_{i}') for i in self.graph.nodes}

        objective = mdl.sum(-x[i] * (1 - x[j]) for i, j in self.graph.edges)
        mdl.minimize(objective)

        self.quadratic_model_theoretical = from_docplex_mp(mdl)

    def generate_CPLEX_model(self):
        """
        Generate the CPLEX model for the MaxCut problem.
        """
        mdl = Model('TFG_MaxCut::CPLEXValidModel')
        x = {i: mdl.binary_var(name=f'x_{i}') for i in self.graph.nodes}
        y = {(i, j): mdl.binary_var(name=f"y_{i}_{j}") for i, j in self.graph.edges}

        objective = mdl.sum(-y[i, j] for i, j in self.graph.edges)
        mdl.minimize(objective)

        for i, j in self.graph.edges:
            mdl.add_constraint(y[(i, j)] <= x[i] + x[j], f"x_{i}*x_{j}_<_2")
            mdl.add_constraint(y[(i, j)] <= 2 - x[i] - x[j], f"x_{i}*x_{j}_>_0")

        self.CPLEX_model = from_docplex_mp(mdl)

    def translate_qp_to_ising(self) -> Tuple[SparsePauliOp, float]:
        """
        Translate the quadratic program to an Ising model.
        
        Returns:
        Tuple[SparsePauliOp, float]: The Ising model and offset.
        """
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(self.quadratic_model)
        return qubo.to_ising()



class TFG_SimplePowerModel_2Partitions(TFG_QuadraticModel):

    def __init__(self, graph: nx.Graph):
        super().__init__()
        self.graph = graph
        self.quadratic_model = None
        self.quadratic_model_theoretical = None
        self.CPLEX_model = None
        self.qubitOp = None
        self.offset = None

        self.nqubits = graph.number_of_nodes()
        self.k = np.mean([self.graph.nodes[i]["weight"] for i in self.graph.nodes])
        self.partitions = 2

        self.generate_models()

    def generate_models(
            self,
            alpha: float = 0.5,
            beta: float = 0.5,
            k: float = None,
            lambda_one_hot: float = None,
            lambda_surplus: float = 0.5,
            withSurplusConstraint: bool = False):
        self.generate_quadratic_model(alpha, beta, k, lambda_one_hot, lambda_surplus)
        self.generate_quadradic_model_theoretical(k, withSurplusConstraint)
        self.generate_CPLEX_valid_model(k)
        self.generate_ising_model(alpha, beta)

    def generate_ising_model(self,
                             alpha: float = None,
                             beta: float = None):

        # The balancing is a sum of all the possible combinations of Z_i, Z_j
        pauli_list = []
        for j in self.graph.nodes:
            for i in range(j):
                # Create arrays for z and x components
               
                z = np.zeros(self.graph.number_of_nodes(), dtype=bool)
                x = np.zeros_like(z, dtype=bool)

                # Set the corresponding positions to True for a Z_i Z_j interaction
                z[i] = True
                z[j] = True

                # Create the Pauli object with the appropriate z and x arrays
                pauli = Pauli((z, x))

                coeff = alpha-beta if (i,j) in self.graph.edges else alpha

                # Append the Pauli object with its coefficient to the list
                pauli_list.append((pauli.to_label(), coeff))


        # Edge discount
        # for every edge we have a ZiZj term as a SparsePauliOp, in negative, because we work the opposite way
        # of MaxCut, we want to avoid cut edges.
        # Then 
        

        # for i, j in self.graph.edges:
        #     # Create arrays for z and x components
        #     z = np.zeros(self.graph.number_of_nodes(), dtype=bool)
        #     x = np.zeros_like(z, dtype=bool)

        #     # Set the corresponding positions to True for a Z_i Z_j interaction
        #     z[i] = True
        #     z[j] = True

        #     # Create the Pauli object with the appropriate z and x arrays
        #     pauli = Pauli((z, x))

        #     # Append the Pauli object with its coefficient to the list
        #     pauli_list.append((pauli.to_label(), -1.0))

        self.qubitOp = SparsePauliOp.from_list(pauli_list)
        self.qubo = self.qubitOp
        self.offset = 0
        

    def generate_quadratic_model(self, 
        alpha: float = 0.5, 
        beta: float = 0.5, 
        k: float = None, # Self-sufficiency parameter
        lambda_one_hot: float = 500.0,
        lambda_surplus: float = 0.5) -> QuadraticProgram:
        """
        Returns a Model from docplex for the following cost functions:

        C_{balanced}(x) = sum_{p}^P ( sum_{n in V_p} x_{i,p} )^2 

        where $x_{i,p}$ is a binary variable that is 1 if node $i$ is in partition $p$ and 0 otherwise. And $V_p$ is the set of nodes in partition $p$.

        $$C_{discount}(x) = sum_{p}^P  ( |E| -sum_{(i,j) in E} x_{i,p} x_{j,p} ) 

        and constraints:

        sum_{p}^P x_{i,p} = 1 for all i in V

        sum_{i}^n x_{i,p} * (w_i - k) <= 0 for all p in P

        """
        self.alpha = alpha
        self.beta = beta
        self.lambda_one_hot = lambda_one_hot
        self.lambda_surplus = lambda_surplus

        # If k is None, we use the average of weights of all graph nodes
        if k is None:
            k = np.mean([self.graph.nodes[i]["weight"] for i in self.graph.nodes])
        
        self.k = k

        self.nqubits = (self.graph.number_of_nodes())

        mdl = Model(name="SimplePowerModel_2Partitions")
        n = self.graph.number_of_nodes()
        E = self.graph.number_of_edges()
        x = {(i): mdl.binary_var(name=f"x_{i}") for i in range(n)}
        # Introduce z_{a,p} variables where a ranges from 0 to K-1 for the inequality constraint
        print("[*] Remember that the auxiliar variables have been ommited.")
        # z = {(a, p): mdl.binary_var(name=f"z_{a}_{p}") for a in range(K) for p in range(P)}


        # Objective function
        C_balanced = (mdl.sum(x[i] for i in range(n)))**2 + (n - mdl.sum(x[i] for i in range(n)))**2

        # C_discount, add cost to cutting edges
        C_discount = mdl.sum(x[i] * (1-x[j]) + (1-x[i])*x[j] for i, j in self.graph.edges)
        # C_discount = E-mdl.sum(2*x[i] * x[j] -x[j] -x[i] for i, j in self.graph.edges)

        # Constraints
            

        # Average surplus constraint
        # for p in range(P):
        #     mdl.add_constraint(mdl.sum(x[i, p] * (self.graph.nodes[i]["weight"]-k) for i in range(n)) <= 0)

        # Expressed as penalty in the paper "Power network optimization: a quantum_approach":
        # c = 0.5 * np.mean([self.graph.nodes[i]["weight"]-k - abs(self.graph.nodes[i]["weight"]-k) for i in self.graph.nodes])

        # Create an array with the weights of each node indexed by the node:
        weights = [self.graph.nodes[i]["weight"] for i in self.graph.nodes]

        # Then the penality becomes:
        # z0 = {(i): mdl.binary_var(name=f"z0_{i}") for i in range(K)}
        # z1 = {(i): mdl.binary_var(name=f"z1_{i}") for i in range(K)}

        # P_surplus = (      ((2 ** K)-0.5)/(-c) * (-c + mdl.sum(x[i]*(weights[i]-k) for i in range(n)))  -mdl.sum(z1[a] * 2**a for a in range(K))        )**2 + \
        #             (      ((2 ** K)-0.5)/(-c) * (-c + mdl.sum((1-x[i])*(weights[i]-k) for i in range(n)))  -mdl.sum(z0[a] * 2**a for a in range(K))        )**2
        
        #P_surplus_simple = (mdl.sum(x[i]*(weights[i]-k) for i in range(n)))**2 + (mdl.sum((1-x[i])*(weights[i]-k) for i in range(n)))**2

        # P_surplus = (mdl.sum(x[i]*weights[i] for i in range(n)) * mdl.sum(x[i] for i in range(n))**-1 -k)*(mdl.sum(x[i]*weights[i] for i in range(n)) * mdl.sum(x[i] for i in range(n))**-1) + \
        #             (mdl.sum((1-x[i])*weights[i] for i in range(n)) * mdl.sum((1-x[i]) for i in range(n))**-1 -k)*(mdl.sum((1-x[i])*weights[i] for i in range(n)) * mdl.sum((1-x[i]) for i in range(n))**-1)

        P_surplus = (mdl.sum(x[i]*(weights[i]-k) for i in range(n)))**2 + (mdl.sum((1-x[i])*(weights[i]-k) for i in range(n)))**2
        print("[*] No surplus constraint has been used")
        # Add the objective function to the model
        print("ALPHA:", self.alpha)
        print("C_BALANCED:",C_balanced)
        print("BETA:", self.beta)
        print("C_DISCOUNT:", C_discount)
        mdl.minimize(self.alpha * C_balanced + self.beta * C_discount + lambda_surplus*P_surplus)
        print("[*] Parabolic surplus has been used")
                     
        

        self.quadratic_model = from_docplex_mp(mdl)
    
    def generate_quadradic_model_theoretical(
        self,
        k: float = None, # Self-sufficiency parameter
        withSurplusConstraint: bool = False) -> QuadraticProgram:
        """
        Returns a Model from docplex with no penalties but the constraints in quadratic mode. This model is NOT convertable to QUBO
        directly if inequalities are present with non-integral coefficients are present:

        C_{balanced}(x) = sum_{p}^P ( sum_{n in V_p} x_{i,p} )^2 

        where $x_{i,p}$ is a binary variable that is 1 if node $i$ is in partition $p$ and 0 otherwise. And $V_p$ is the set of nodes in partition $p$.

        $$C_{discount}(x) = sum_{p}^P  ( |E| -sum_{(i,j) in E} x_{i,p} x_{j,p} ) 

        and constraints:

        sum_{p}^P x_{i,p} = 1 for all i in V

        sum_{i}^n x_{i,p} * (w_i - k) <= 0 for all p in P


        """

        # If k is None, we use the average of weights of all graph nodes
        if k is None:
            k = np.mean([self.graph.nodes[i]["weight"] for i in self.graph.nodes])

        mdl = Model(name="SimplePowerModel_2Partitions_Theoretical")
        n = self.graph.number_of_nodes()
        E = self.graph.number_of_edges()
        x = {(i): mdl.binary_var(name=f"x_{i}") for i in range(n)}


        # Objective function
        C_balanced = (mdl.sum(x[i] for i in range(n)))**2 + (n - mdl.sum(x[i] for i in range(n)))**2

        # # C_discount
        # C_discount = 2*E-mdl.sum(x[i] * x[j] + (1-x[i]) * (1-x[j]) for i, j in self.graph.edges)
        # C_discount, add cost to cutting edges
        C_discount = mdl.sum(x[i] * (1-x[j]) + (1-x[i])*x[j] for i, j in self.graph.edges)


        # Create an array with the weights of each node indexed by the node:
        weights = [self.graph.nodes[i]["weight"] for i in self.graph.nodes]
        
        # P_surplus = (mdl.sum(x[i]*weights[i] for i in range(n)) * mdl.sum(x[i] for i in range(n))**-1 -k)*(mdl.sum(x[i]*weights[i] for i in range(n)) * mdl.sum(x[i] for i in range(n))**-1) + \
        #             (mdl.sum((1-x[i])*weights[i] for i in range(n)) * mdl.sum((1-x[i]) for i in range(n))**-1 -k)*(mdl.sum((1-x[i])*weights[i] for i in range(n)) * mdl.sum((1-x[i]) for i in range(n))**-1)

        P_surplus = (mdl.sum(x[i]*(weights[i]-k) for i in range(n)))**2 + (mdl.sum((1-x[i])*(weights[i]-k) for i in range(n)))**2
        print("[*] No surplus constraint has been used")
        # Average surplus constraint
        if withSurplusConstraint:
            mdl.add_constraint(mdl.sum(x[i] * (self.graph.nodes[i]["weight"]-k) for i in range(n)) <= 0,'Group 0 surplus')
            mdl.add_constraint(mdl.sum((1-x[i]) * (self.graph.nodes[i]["weight"]-k) for i in range(n)) <= 0, 'Group 1 surplus')


        # Add the objective function to the model
        mdl.minimize(C_balanced + C_discount)        

        self.quadratic_model_theoretical = from_docplex_mp(mdl)


    def generate_CPLEX_valid_model(self,
                                   k: float = None):

        # If k is None, we use the average of weights of all graph nodes
        if k is None:
            k = np.mean([self.graph.nodes[i]["weight"] for i in self.graph.nodes])

        mdl = Model(name="SimplePowerModel_2Partitions_Theoretical")
        n = self.graph.number_of_nodes()
        E = self.graph.number_of_edges()
        x = {(i): mdl.binary_var(name=f"x_{i}") for i in range(n)}


        # Objective function
        C_balanced = (mdl.sum(x[i] for i in range(n)))**2 + (n - mdl.sum(x[i] for i in range(n)))**2

        # C_discount
        # C_discount = 2*E-mdl.sum(x[i] * x[j] + (1-x[i]) * (1-x[j]) for i, j in self.graph.edges)
        # It is non convex, so must be rewriten in a convex form using auxiliary variables
        y = {(i,j): mdl.binary_var(name=f"y_{i}_{j}") for i, j in self.graph.edges}

        # We need to penalize (sum) the edges that are not used (their vertices belong to different partitions)
        C_discount = mdl.sum(y[i,j] for i, j in self.graph.edges)

        for i, j in self.graph.edges:
            mdl.add_constraint(y[(i,j)] <= x[i] + x[j], f"x_{i}*x_{j}_<_2")
            mdl.add_constraint(y[(i,j)] <= 2 - x[i] - x[j], f"x_{i}*x_{j}_>_0")

        mdl.add_constraint(mdl.sum(x[i] * (self.graph.nodes[i]["weight"]-k) for i in range(n)) <= 0)
        mdl.add_constraint(mdl.sum((1-x[i]) * (self.graph.nodes[i]["weight"]-k) for i in range(n)) <= 0)



        # Add the objective function to the model
        mdl.minimize(C_balanced + C_discount)        

        self.CPLEX_model = from_docplex_mp(mdl)

    def translate_qp_to_ising(self) -> Tuple[SparsePauliOp, float]:
        """
        Translate the quadratic program to an Ising model.
        
        Returns:
        Tuple[SparsePauliOp, float]: The Ising model and offset.
        """
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(self.quadratic_model)
        return qubo.to_ising()
    

    def to_dict(self):
        return {
            'graph': nx.node_link_data(self.graph),
            'quadratic_model': self.quadratic_model.export_as_lp_string() if self.quadratic_model else None,
            'quadratic_model_theoretical': self.quadratic_model_theoretical.export_as_lp_string() if self.quadratic_model_theoretical else None,
            'CPLEX_model': self.CPLEX_model.export_as_lp_string() if self.CPLEX_model else None,
            'qubitOp': self.qubitOp.to_list() if self.qubitOp else None,
            'offset': self.offset,
            'nqubits': self.nqubits,
            'k': self.k,
            'partitions': self.partitions
        }

    @classmethod
    def from_dict(cls, data):
        graph = nx.node_link_graph(data['graph'])
        instance = cls(graph)
        instance.quadratic_model = QuadraticProgram().from_lp_string(data['quadratic_model']) if data['quadratic_model'] else None
        instance.quadratic_model_theoretical = QuadraticProgram().from_lp_string(data['quadratic_model_theoretical']) if data['quadratic_model_theoretical'] else None
        instance.CPLEX_model = QuadraticProgram().from_lp_string(data['CPLEX_model']) if data['CPLEX_model'] else None
        instance.qubitOp = SparsePauliOp.from_list(data['qubitOp']) if data['qubitOp'] else None
        instance.offset = data['offset']
        instance.nqubits = data['nqubits']
        instance.k = data['k']
        instance.partitions = data['partitions']
        return instance
