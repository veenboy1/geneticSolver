import random
import networkx as nx
import math
import parameters as p
import network_functions as nf
from collections import defaultdict


# ---------- Fitness function creation ---------- #
# Creating a subgraph from the chromosome
def create_subgraph_from_chromosome(G, chromosome, verbose=False):
    # Create a new directed graph for the subgraph
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(G.nodes())

    # Iterate through the edges of the original graph along with the chromosome values
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        # If the corresponding chromosome value is 1, add the edge to the subgraph
        if chromosome[i] == 1:
            subgraph.add_edge(u, v, **data)
            if verbose:
                print(f'Edge {i}: ({u}, {v}) {data}')

    return subgraph


# Fitness function for directed graphs (with testing and debugging)
def fitness(chromosome, G, demand, verbose=False, subgraph_name=None):
    # creating some variables
    F = 0

    # generate the subgraph
    subgraph = create_subgraph_from_chromosome(G, chromosome, verbose=verbose)

    # function f(G_s)
    xf = sum(data[p.cost_attribute] for u, v, data in subgraph.edges(data=True))
    f = math.exp(-p.sigma_f * xf)

    if verbose:
        print(f'Sum cost of this subgraph: {xf}')

    # calculating the value of some variables
    d_s = 0
    c_s1 = 0
    c_s2 = 0

    for i in range(len(demand)):
        o, d = demand[i]  # Extract origin and destination from array
        try:
            # This try/except block will detect if every (o, d) pair is connected.
            # Each failure adds to the d_s (num of disconnections) AND the c_s1/c_s2.
            sp_dist_s = nx.shortest_path_length(subgraph, o, d)
            sp_dist = nx.shortest_path_length(G, o, d)
            # TODO: ask Mike or someone who knows better if this should be continuous or discrete
            if sp_dist - sp_dist_s > p.gamma_abs + .001:
                # determines if the B7 constraint is violated
                if verbose:
                    print('Failed B7')
                c_s2 += 1
            if sp_dist_s / sp_dist > p.gamma_rel - .001:
                # determines if B6 constraint is violated
                if verbose:
                    print('Failed B6')
                c_s1 += 1
        except nx.exception.NetworkXNoPath:
            if verbose:
                print('Uh oh... somebody lost connection... ')
            d_s += 1
            c_s1 += 1
            c_s2 += 1

    if verbose:
        if subgraph_name:
            print(f'Completed inspection of subgraph {subgraph_name}.\nCalculating the value of F now.')
        else:
            print(f'Completed inspection of subgraph.\nCalculating the value of F now.')

    # calculating g(d_s)
    g = math.exp(-p.sigma_g * d_s)

    # calculating h(c_s1, c_s2)
    h = 0.5 * (math.exp(-p.sigma_h1 * c_s1) + math.exp(-p.sigma_h2 * c_s2))

    # finding the value of F
    F += f
    F += g
    F += h

    # reporting the score of the subgraph:
    if verbose:
        if subgraph_name:
            print(f'For subgraph {subgraph_name}')
        print(f'F: {F:.2f}\nf: {f:.2f}\ng: {g:.2f}\nh: {h:.2f}')

    return F


# Fitness function for PyGAD implementation
def slim_fit(ga_instance, chromosome, chromosome_idx):
    return float(fitness(chromosome, nf.G, p.speedy_demand))


# ---------- Creating the initial population ---------- #
# creates a random chromosome
def random_chromosome(G):
    return [random.choice([0, 1]) for _ in range(len(G.edges()))]


# creates a chromosome that includes the entire graph
def whole_chromosome(G):
    return [1 for i in range(len(G.edges()))]


# creates a chromosome from the shortest paths of all the OD pairs
def create_chromosome_from_shortest_paths(G, demand):
    # Initialize a dictionary to keep track of whether an edge is in any shortest path
    edge_in_path = {edge: 0 for edge in G.edges()}

    # Loop through each origin-destination pair
    for o, d in demand:
        # Find the shortest path between origin and destination
        shortest_path = nx.shortest_path(G, source=o, target=d, weight='length')

        # Convert the shortest path to a list of edges
        path_edges = list(zip(shortest_path[:-1], shortest_path[1:]))
        # print(path_edges)
        # Mark these edges as present in the chromosome
        for edge in path_edges:
            edge_in_path[edge] = 1

    # Convert the edge presence dictionary to a list of 1s and 0s representing the chromosome
    chromosome = [edge_in_path[edge] for edge in G.edges()]

    return chromosome


def simple_greedy_path_optimizer(graph, demand):
    # Sort edges by length (cost)
    sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2][p.cost_attribute])

    # Group edges by length to handle ties
    grouped_edges = {}
    for edge in sorted_edges:
        length = edge[2][p.cost_attribute]
        if length not in grouped_edges:
            grouped_edges[length] = []
        grouped_edges[length].append(edge)

    # Create a subgraph to build our solution
    solution_graph = nx.DiGraph()
    solution_graph.add_nodes_from(graph.nodes())

    # NEW: Convert demand_array to a set of tuples for faster lookup
    demand_set = set(map(tuple, demand))

    # NEW: Function to check if all demands are satisfied
    def all_demands_satisfied():
        for origin, destination in demand_set:
            if not nx.has_path(solution_graph, origin, destination):
                return False
        return True

    # Add edges until all demands are satisfied
    while not all_demands_satisfied():
        # Get the group of edges with the lowest cost
        min_length = min(grouped_edges.keys())
        edges_group = grouped_edges[min_length]

        # Randomly select an edge from this group (breaking ties randomly)
        edge = random.choice(edges_group)
        start, end, data = edge

        # Add the edge to our solution
        solution_graph.add_edge(start, end, **data)

        # Remove this edge from our grouped_edges
        grouped_edges[min_length].remove(edge)
        if not grouped_edges[min_length]:
            del grouped_edges[min_length]

    return solution_graph


# Creates a chromosome using a basic greedy algorithm
def create_greedy_chromosomes(G, demand, num_copies):

    # generate the chromosomes
    chromosomes = []
    for i in range(num_copies):
        # building the greedy graph
        greedy = simple_greedy_path_optimizer(G, demand)

        # creating the chromosome
        chromosome = []
        for edge in G.edges:
            if edge in greedy.edges:
                chromosome.append(1)
            else:
                chromosome.append(0)

        chromosomes.append(chromosome)

    return chromosomes


# --- Generates the initial population --- #
def generate_initial_population(G, demand, idea=None):
    if idea == '1':
        # a bunch of variations on the shortest path chromosome
        sp_chromosome = create_chromosome_from_shortest_paths(G, demand)
        gene_pool = initial_population_mutation(sp_chromosome, p.num_copies)
    elif idea == '2':
        # some of the shortest path chromosomes, some of the greedy algorithm chromosomes
        gene_pool = []
        sp_chromosome = create_chromosome_from_shortest_paths(G, demand)
        gene_pool += initial_population_mutation(sp_chromosome, p.num_copies_each_type)
        gene_pool += create_greedy_chromosomes(G, demand, p.num_copies_each_type)
    else:
        gene_pool = [whole_chromosome(G),
                     create_chromosome_from_shortest_paths(G, demand)]

        for i in range(2):
            mutants = initial_population_mutation(gene_pool[i], 24)
            gene_pool = gene_pool + mutants

    return gene_pool


# ---------- Mutation functions ---------- #
def initial_population_mutation(chromosome, num_copies, mutation_rate=p.initial_mutt_rate):
    mutated_population = []

    for _ in range(num_copies):
        # Create a copy of the original chromosome to mutate
        mutated_chromosome = chromosome.copy()

        # Iterate through each gene in the chromosome
        for i in range(len(mutated_chromosome)):
            if random.random() < mutation_rate:
                # Flip the gene (0 -> 1 or 1 -> 0)
                mutated_chromosome[i] = 1 - mutated_chromosome[i]

        # Add the mutated chromosome to the population list
        mutated_population.append(mutated_chromosome)

    return mutated_population


def custom_mutation(offspring, ga_instance):
    # NOTE: Currently not doing this. I think that inversion mutation should
    # do everything that we need to do.
    # Iterate through each chromosome
    for chromosome in offspring:
        # Iterate through each gene in the chromosome
        for gene_idx in range(len(chromosome)):
            # If the gene is not 0 or 1, replace it with a random choice of 0 or 1
            if chromosome[gene_idx] not in [0, 1]:
                chromosome[gene_idx] = random.choice([0, 1])

    return offspring
