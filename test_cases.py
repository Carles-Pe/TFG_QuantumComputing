import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import copy
import pandas as pd
import cartopy
import cartopy.crs as ccrs



class TFG_ITestCase():
    def __init__(self) -> None:
        self.generate_cases()

    def generate_cases(self) -> None:
        raise ValueError("generate_cases() method not implemented")

    def select_case(self,
                    key: str = None) -> None:
        raise ValueError("select_case() method not implemented")

    def plot_case(self,
                  G: nx.Graph,
                  solution: np.ndarray = None) -> None:
        raise ValueError("plot_case() method not implemented")


class TFG_PowerDistributionExamples(TFG_ITestCase):
    
    def __init__(self) -> None:
        super().__init__()
        
        return
    
    def generate_cases(self) -> None:
        print("Generating power distribution examples")
        self.generate_fully_connected()
        self.generate_cliques()
        print("Power distribution examples generated")
        return
    
    def select_case(self,
                    key: str = None) -> None:
        if key == "fc_4":
            return self.fc_4
        elif key == "fc_6":
            return self.fc_6
        elif key == "fc_8":
            return self.fc_8
        elif key == "fc_10":
            return self.fc_10
        elif key == "clique_6":
            return self.clique_6
        elif key == "clique_8":
            return self.clique_8
        else:
            # Attempt to parse key as 'fc_classic_x' where x is the number of nodes
                if key.startswith("fc_classic_"):
                        n = int(key.split("_")[-1])
                        return self.fc_generator_classic(n)
                else:
                      raise ValueError("Key not found")
        return
    
    def plot_case(self,
                G: nx.Graph,
                solution: np.ndarray = None,
                store_in_folder: str = None
                ) -> None:
        self.draw_graph(G, solution)
        return

    def fc_generator_classic(self, n: int) -> nx.Graph:
        # No randomness
        G = nx.Graph()
        # weight is 5.0 + random number from 0 to 0.5


        nlist = [(i, {"weight": 5.0+np.round(np.random.random()*0.5,decimals=2)}) for i in range(n//2)]
        nlist += [(n//2, {"weight": 5.0*(n//2) + np.round(np.random.random()*0.5*(n//2),decimals=2)})]
        nlist += [(i, {"weight": np.round(np.random.random()*0.5,decimals=2)}) for i in range(n//2+1, n)]
        G.add_nodes_from(nlist)

        elist = [(i, j) for i in range(n) for j in range(i+1, n)]
        G.add_edges_from(elist)

        return G

#     def random_fc_generator(self, n: int, seed: int) -> nx.Graph:
#         G = nx.Graph()
#         np.random.seed(seed)
#         partition_points = np.sort(np.random.random(n-1) * 5*n)
    

    def generate_fully_connected(self):
        self.fc_4 = nx.Graph() # Fully connected graph with 4 nodes
        self.fc_6 = nx.Graph() # Fully connected graph with 6 nodes
        self.fc_8 = nx.Graph() # Fully connected graph with 8 nodes
        self.fc_10 = nx.Graph() # Fully connected graph with 10 nodes

        nlist_4 = [(0, {"weight": 5.0}), (1, {"weight": 5.0}), (2, {"weight": 10.0}), (3, {"weight": 0.1})]

        nlist_6 = [(0, {"weight": 5.0}), (1, {"weight": 5.0}), (2, {"weight": 5.0}), 
           (3, {"weight": 0.1}), (4, {"weight": 15.0}), (5, {"weight": 0.3})]

        nlist_8 = [(0, {"weight": 5.0}), (1, {"weight": 5.0}), (2, {"weight": 5.0}), (3, {"weight": 5.0}),
                (4, {"weight": 0.1}), (5, {"weight": 20.0}), (6, {"weight": 0.3}), (7, {"weight": 0.0})]


        nlist_10 = [(0, {"weight": 5.0}), (1, {"weight": 5.0}), (2, {"weight": 5.0}), (3, {"weight": 5.0}), (4, {"weight": 5.0}),
                (5, {"weight": 0.1}), (6, {"weight": 25.0}), (7, {"weight": 0.3}), (8, {"weight": 0.0}), (9, {"weight": 0.1})]

        

        self.fc_4.add_nodes_from(nlist_4)
        self.fc_6.add_nodes_from(nlist_6)
        self.fc_8.add_nodes_from(nlist_8)
        self.fc_10.add_nodes_from(nlist_10)


        elist_4 = [(0,1), (0,2), (0,3),
                (1,2), (1,3),
                (2,3)]
        

        # Fully connected graph:
        elist_6 = [(0,1), (0,2), (0,3), (0,4), (0,5),
                (1,2), (1,3), (1,4), (1,5),
                (2,3), (2,4), (2,5),
                (3,4), (3,5),
                (4,5)]


        elist_8 = [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7),
                (1,2), (1,3), (1,4), (1,5), (1,6), (1,7),
                (2,3), (2,4), (2,5), (2,6), (2,7),
                (3,4), (3,5), (3,6), (3,7),
                (4,5), (4,6), (4,7),
                (5,6), (5,7),
                (6,7)]

        elist_10 = [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9),
                (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9),
                (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9),
                (3,4), (3,5), (3,6), (3,7), (3,8), (3,9),
                (4,5), (4,6), (4,7), (4,8), (4,9),
                (5,6), (5,7), (5,8), (5,9),
                (6,7), (6,8), (6,9),
                (7,8), (7,9),
                (8,9)]
        
        self.fc_4.add_edges_from(elist_4)
        self.fc_6.add_edges_from(elist_6)
        self.fc_8.add_edges_from(elist_8)
        self.fc_10.add_edges_from(elist_10)


    def generate_cliques(self):

        # These graphs tests the disocunt edge cost

        self.clique_6 = nx.Graph()
        self.clique_8 = nx.Graph()


        nlist_6 = [(0, {"weight": 0.1}), (1, {"weight": 0.5}), (2, {"weight": 0.3}), 
           (3, {"weight": 0.1}), (4, {"weight": 0.5}), (5, {"weight": 0.3})]

        nlist_8 = [(0, {"weight": 0.1}), (1, {"weight": 0.5}), (2, {"weight": 0.3}), (3, {"weight": 0.0}),
                (4, {"weight": 0.1}), (5, {"weight": 0.5}), (6, {"weight": 0.3}), (7, {"weight": 0.0})]

        self.clique_6.add_nodes_from(nlist_6)
        self.clique_8.add_nodes_from(nlist_8)

        

        elist_6 = [(0,1), (0,2), (1,2), 
                (3,4), (3,5), (4,5),
                (2,3)] # Connect the two fully-connected subgraphs


        elist_8 = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3), 
                (4,5), (4,6), (4,7), (5,6), (5,7), (6,7),
                (3,4)] # Connect the two fully-connected subgraphs
        
        self.clique_6.add_edges_from(elist_6)
        self.clique_8.add_edges_from(elist_8)
        return


    def draw_graph(self, G: nx.Graph, solution: np.ndarray = None, store_in_folder: str = None):
        color_map = []
        if solution is not None:
            partition_colors = [
                        '#FF5733',  # Bright red
                        '#33FF57',  # Bright green
                        '#3357FF',  # Bright blue
                        '#F3FF33',  # Bright yellow
                        '#F333FF',  # Bright magenta
                        '#33FFF3',  # Cyan
                        '#FF33F3',  # Pink
                        '#5733FF'   # Violet
                ]
            color_map = [] 
            for x in solution:
                if x > 0.5:
                        color_map.append(partition_colors[0])
                else:
                        color_map.append(partition_colors[1])
            
        
        else:
            color_map = ["r" for node in G.nodes()]
        
        pos = nx.spring_layout(G)
        default_axes = plt.axes(frameon=True)
        nx.draw_networkx(G, node_color=color_map, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

        # Labels can be customized here if needed, for now, it uses node indices
        for node, (x, y) in pos.items():
                if "weight" in G.nodes[node]: plt.text(x, y+0.1, s=f'Weight: {G.nodes[node]["weight"]}', bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center')

        if store_in_folder is not None:
            plt.savefig(store_in_folder+"/graph.png")

        

class TFG_ScandinavianPowerDistribution(TFG_ITestCase):

    def __init__(self) -> None:
        super().__init__()

        return
    
    # @classmethod
    # def generate_seeded(self,
    #                     seed: int = None) -> None:
    #     self.generate_scandinavian_11(seed)
    #     return
    

    
    def generate_cases(self) -> None:
        print("Generating scandinavian power distribution")
        self.generate_scandinavian_11()
        print("Scandinavian power distribution generated")
        return


    # def generate_cases(self, seed: int = None):
    #     self.generate_scandinavian_11(seed)

    def select_case(self,
                    key: str = None) -> nx.Graph:
        if key == "scandinavian_11":
            return self.scandinavian_11
        elif key == "scandinavian_4":
            return self.scandinavian_4
        elif key == "scandinavian_5":
            return self.scandinavian_5
        elif key == "scandinavian_6":
            return self.scandinavian_6
        elif key == "scandinavian_7":
            return self.scandinavian_7
        elif key == "scandinavian_8":
            return self.scandinavian_8
        elif key == "scandinavian_9":
            return self.scandinavian_9
        elif key == "scandinavian_10":
            return self.scandinavian_10
        else:
            raise ValueError("Key not found")

    def plot_case(self,
                    G: nx.Graph,
                    solution: np.ndarray = None,
                    store_in_folder: str = None) -> None:
        self.draw_map(G, solution, store_in_folder=store_in_folder)
        return

    def generate_scandinavian_11(self, seed: int = None):
        print("Generating scandinavian 11 graph")

        if seed is not None:
            np.random.seed(seed)

        G = nx.Graph()

        nodes_coords = pd.read_csv('data/11_nodes_coords.csv', delimiter=';', on_bad_lines='error')
        edges_coords = pd.read_csv('data/11_nodes_links.csv', delimiter=';')

        for i, row in nodes_coords.iterrows():
            if row is not None:
                weight = np.round(np.random.random(),decimals=3)
                G.add_node(i, name=row['name'] ,pos=(row['x'], row['y']), weight=weight)



        # Add edges, from a csv in matrix form (11x11) that contains 1 if (i,j) is an edge, 0 otherwise
        for i, row in edges_coords.iterrows():
            for j, val in enumerate(row):
                if j == 0:
                    continue
                elif val == 1:
                    G.add_edge(i, j-1)

        self.scandinavian_11 = G

        scandinavian_x = []
        # Generate different self.scandinavian_x keeping x of the lower index nodes:
        for i in range(4, 11):
            G = copy.deepcopy(self.scandinavian_11)
            for j in range(i, 11):
                G.remove_node(j)
            scandinavian_x.append(G)

        self.scandinavian_4 = scandinavian_x[0]
        self.scandinavian_5 = scandinavian_x[1]
        self.scandinavian_6 = scandinavian_x[2]
        self.scandinavian_7 = scandinavian_x[3]
        self.scandinavian_8 = scandinavian_x[4]
        self.scandinavian_9 = scandinavian_x[5]
        self.scandinavian_10 = scandinavian_x[6]



        


    def draw_map(self, G: nx.Graph, solution: np.array = None, store_in_folder: str = None):

        color_map = []   

        if solution is not None:
            # 8 different colors for 8 different partitions:
            partition_colors = [
                '#FF5733',  # Bright red
                '#33FF57',  # Bright green
                '#3357FF',  # Bright blue
                '#F3FF33',  # Bright yellow
                '#F333FF',  # Bright magenta
                '#33FFF3',  # Cyan
                '#FF33F3',  # Pink
                '#5733FF'   # Violet
            ]
             

            for x in solution:
                if x > 0.5:
                    color_map.append(partition_colors[0])
                else:
                    color_map.append(partition_colors[1])

        else :
            color_map = ['red' for node in G.nodes()]


        pos = nx.get_node_attributes(G, 'pos')
        
        lon_min = min(lon for lon, lat in pos.values()) - 1  # padding of 1 degree
        lon_max = max(lon for lon, lat in pos.values()) + 1
        lat_min = min(lat for lon, lat in pos.values()) - 1
        lat_max = max(lat for lon, lat in pos.values()) + 1

        # Setup plot with a geographic projection
        fig, ax = plt.subplots(
            figsize=(10, 10),
            subplot_kw={'projection': ccrs.PlateCarree()})
        #ax.set_global()
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])

        # Add features to the map
        ax.add_feature(cartopy.feature.LAND)
        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')

        # Draw lines for edges using great circle (more geographically accurate)
        for edge in G.edges():
            start_pos = pos[edge[0]]
            end_pos = pos[edge[1]]
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                    color='blue', linewidth=2, marker='o', transform=ccrs.Geodetic())

        # Draw nodes
        for node, (lon, lat) in pos.items():
            ax.plot(lon, lat, 'o', markersize=5, color=color_map[node], transform=ccrs.Geodetic())
            ax.text(lon, lat, G.nodes()[node]['name'], transform=ccrs.Geodetic(), fontsize=12, ha='right')
            ax.text(lon, lat, f'{G.nodes()[node]["weight"]}', transform=ccrs.Geodetic(), fontsize=8, ha='left')

        # Set title and display the plot
        ax.set_title('World Map with Graph Overlay')
        if store_in_folder is not None:
            plt.savefig(store_in_folder+"/map.png")
        plt.show()


        return
    
    def draw_coloured_map(self, G: nx.Graph, solution: np.array):

        # 8 different colors for 8 different partitions:
        partition_colors = [
            '#FF5733',  # Bright red
            '#33FF57',  # Bright green
            '#3357FF',  # Bright blue
            '#F3FF33',  # Bright yellow
            '#F333FF',  # Bright magenta
            '#33FFF3',  # Cyan
            '#FF33F3',  # Pink
            '#5733FF'   # Violet
        ]
        color_map = []    

        for x in solution:
            if x > 0.5:
                color_map.append(partition_colors[0])
            else:
                color_map.append(partition_colors[1])


class TFG_TestCaseSampler():
    '''
    Usage: 
    examples1 = TFG_TestCaseSampler(<test_case_code>).get_test_case()

    Provides the test case for the desired code. All of the test cases follow the TestCase "interface"
    '''
    def __init__(self, 
                 testCaseKey: str = None) -> None:
        if testCaseKey == "scandinavia":
            self.testCase = TFG_ScandinavianPowerDistribution()
        else:
            self.testCase = TFG_PowerDistributionExamples()

    def get_test_case(self) -> TFG_ITestCase:
        return self.testCase
        
