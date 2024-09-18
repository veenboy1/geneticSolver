import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import parameters as p
from genetic_functions import create_subgraph_from_chromosome

# ---------- Load network data ---------- #
def read_node_pos_geojson(file_name, verbose=False):
    gdf = gpd.GeoDataFrame.from_file(file_name)

    if verbose:
        print(gdf)

    positions = {row['id']: (row.geometry.y, row.geometry.x) for _, row in gdf.iterrows()}

    if verbose:
        print(positions)

    return positions


def read_network_info(file_name, skip_rows=7, returndf=False, tilde='~ ',
                      init_node='Init node ', term_node='Term node ',
                      is_anaheim=False):
    df = pd.read_csv(file_name, sep='\t', skiprows=skip_rows)
    nodes = list(set(df[init_node].to_list() + df[term_node].to_list()))
    if is_anaheim:
        df['length'] = df['length'] / 5280
    edge_params = df.drop(columns=[tilde, init_node, term_node, ';']).to_dict(orient='records')
    edges = list(zip(df[init_node].values, df[term_node].values, edge_params))

    if returndf:
        return nodes, edges, df
    else:
        return nodes, edges


def create_disneytown(dg=False, save_file=None):
    G = nx.DiGraph()

    nodes2, edges = read_network_info(p.ana_net_file, 8, tilde='~',
                                      init_node='init_node', term_node='term_node',
                                      is_anaheim=True)

    nodes = list(read_node_pos_geojson(p.ana_node_file).keys())

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    if dg:
        positions = read_node_pos_geojson(p.ana_node_file)
        plt.figure(figsize=p.fig_size)
        nx.draw(G, pos=positions, with_labels=True, node_size=200, node_color="lightblue")
        if save_file:
            plt.savefig(save_file, dpi=p.dpi)
            plt.close()
        else:
            plt.show()

    return G


def create_test_net(dg=False):
    """
    Creates a 5x5 directed grid graph with bidirectional edges and a diagonal path.

    This function constructs a directed graph with nodes numbered from 1 to 25 arranged in a 5x5 grid.
    The graph includes bidirectional edges between adjacent nodes (right and down), as well as a
    one-directional diagonal path from node 1 to 25.

    Parameters:
    draw (bool): If True, the function will use matplotlib to draw and display the graph with nodes
                 positioned in a grid layout. Default is False.

    Returns:
    G (networkx.DiGraph): The created directed graph with the specified node and edge configurations.
    """

    G = nx.DiGraph()

    # Just trying to make a 5x5 grid
    grid_size = 5
    nodes = range(1, grid_size ** 2 + 1)
    G.add_nodes_from(nodes)

    # Right and Left edges
    G.add_edges_from([(i, i + 1, {'length': 1}) for i in range(1, grid_size ** 2) if i % grid_size != 0])  # Right
    G.add_edges_from([(i + 1, i, {'length': 1}) for i in range(1, grid_size ** 2) if i % grid_size != 0])  # Left

    # Down and Up edges
    G.add_edges_from([(i, i + grid_size, {'length': 1}) for i in range(1, grid_size ** 2 - grid_size + 1)])  # Down
    G.add_edges_from([(i + grid_size, i, {'length': 1}) for i in range(1, grid_size ** 2 - grid_size + 1)])  # Up
    # Add one-directional diagonal from node 1 -> 7 -> 13 -> 19 -> 25
    diagonal_nodes = [1, 7, 13, 19, 25]
    G.add_edges_from((diagonal_nodes[i], diagonal_nodes[i + 1], {'length': 1}) for i in range(len(diagonal_nodes) - 1))

    if dg:
        # Generate positions for a grid layout
        pos = {i: ((i - 1) % grid_size, (i - 1) // grid_size) for i in range(1, grid_size ** 2 + 1)}

        # Draw the graph with the calculated grid positions
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='#aaaaff', arrowsize=20)
        edge_labels = nx.get_edge_attributes(G, 'length')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title('5x5 Grid Graph in Grid Layout')
        plt.show()

    return G


def create_sioux_falls(draw_net=False, num_of_drawings=1):
    # initialize graph
    G = nx.DiGraph()

    # add nodes and edges
    nodes, edges = read_network_info(p.sf_net_file)
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G


# ---------- Reporting the results of the GA run ---------- #
def read_node_pos(file_name, verbose=False):
    df = pd.read_csv(file_name, sep='\t')
    if verbose:
        print(df)

    # might need to change 'Node' to something else if not Sioux falls
    nodes_position_dict = df.set_index('Node')[['X', 'Y']].apply(tuple, axis=1).to_dict()
    if verbose:
        print(nodes_position_dict)

    return nodes_position_dict


def report_results(best_solution, location=None, graph=None, title='Geoff is the best'):
    if graph is None:
        network1 = create_subgraph_from_chromosome(G, best_solution, True)
    else:
        network1 = graph

    plt.figure(figsize=p.fig_size)

    # Determining the locations of nodes in the network
    if location == 'test net':
        # Graphs the best solution
        positions = {i: ((i - 1) % p.grid_size, (i - 1) // p.grid_size)
                     for i in range(1, p.grid_size ** 2 + 1)}
        nx.draw_networkx_nodes(network1, pos=positions, node_size=45, node_color='lightblue')
        nx.draw_networkx_labels(network1, pos=positions, font_size=10, font_color="black")
        nx.draw_networkx_edges(network1, pos=positions, edge_color='red', alpha=0.9)
    elif location == 'sioux falls':
        positions = read_node_pos(p.sf_node_file)
    else:
        print('No positions available for this location (Geoff didn\'t do it yet)')
        positions = None

    # Creating plotting the network if all information is available
    if positions and location is not None:
        plt.figure(figsize=p.fig_size)

        # uncomment if you want to change the color of the nodes 1-38 (ones with demand)
        node_colors = ["red" if 1 <= node <= 38 else "lightblue" for node in network1.nodes()]
        # node_colors = ['lightblue' for _ in network1.nodes()]

        nx.draw_networkx_nodes(network1, pos=positions, node_size=45, node_color=node_colors)
        nx.draw_networkx_labels(network1, pos=positions, font_size=10, font_color="black")
        nx.draw_networkx_edges(network1, pos=positions, edge_color='red', alpha=0.9)

        # additional things for the plot
        plt.title(str(title))

        plt.show()
    else:
        print('Plot not shown.')

    return network1


# ---------- Cleaning off any unused edges in the graph ---------- #
def prune_dead_branches(G, pairs):
    # Initialize a set to track edges that are part of any shortest path
    shortest_path_edges = set()

    # Iterate over all origin-destination pairs
    for o, d in pairs:
        try:
            # Find the shortest path between origin and destination
            shortest_path = nx.shortest_path(G, source=o, target=d, weight=p.cost_attribute,
                                             method='dijkstra')

            # Extract edges from this path and add them to the set
            path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
            shortest_path_edges.update(path_edges)

        except nx.NetworkXNoPath:
            # If no path exists between the nodes, skip this pair
            continue

    # Create a copy of the graph to prune
    pruned_graph = G.copy()

    # Iterate over all edges in the original graph
    for edge in list(G.edges()):
        # If an edge is not part of any shortest path, remove it from the graph
        if edge not in shortest_path_edges:
            pruned_graph.remove_edge(*edge)

    report_results(None, location='sioux falls', graph=pruned_graph, title='Pruned Graph')

    # Display pruned solution
    xf = sum(data[p.cost_attribute] for u, v, data in pruned_graph.edges(data=True))

    print("\n" + "=" * 40)
    print("🌟 PRUNED GRAPH COST SUMMARY 🌟")
    print("=" * 40)
    print(f"""
        💰 Total Cost: {xf:.2f}
    """)
    print("=" * 40 + "\n")

    return pruned_graph


# Graph variable
G = create_sioux_falls()
