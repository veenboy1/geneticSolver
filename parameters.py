from numpy import array
from itertools import product
# Parameters for the Genetic Algorithm

# ----------- Things you must change to change location ----------- #
# Change the G variable in network_functions
# Change the cost_attribute below
# Change the location parameter below (Anaheim, sioux falls)

location = 'Anaheim'

population_size = 40  # Number of chromosomes in the population
cost_attribute = 'length'  # 'Length '  # Edge attribute to use for calculating cost

# ----------- Network File Paths ----------- #
ana_node_file = './Anaheim/anaheim_nodes.geojson'
ana_net_file = './Anaheim/Anaheim_net.txt'
sf_net_file = './SiouxFalls/SiouxFalls_net.txt'
sf_node_file = './SiouxFalls/SiouxFalls_node.txt'

# ----------- Other non-essential parameters ----------- #
fig_size = (20, 20)
dpi = 200
grid_size = 5
display_edge_weights = False

# ----------- Fitness function constants ----------- #
sigma_f = .1       # for test net, .25 works well; .05 for Sioux Falls
sigma_g = .25
sigma_h1 = .25
sigma_h2 = .25

# ----------- Detour Maximums ----------- #
gamma_abs = 50
gamma_rel = 1.5


# ----------- Demand sets ----------- #
def create_many_to_many_demand(origins, destinations, demand=100):
    # this will really quickly create a demand dictionary with equal demand to all
    # origin and destination pairs specified from lists of origin and destination nodes
    return {(a, b): demand for a, b in product(origins, destinations) if not (a == b)}


origins = [1, 35]
destinations = [4, 7]
dummy_demand = create_many_to_many_demand(origins, destinations)

speedy_demand = array(list(dummy_demand.keys()))

# ----------- GA Parameters ----------- #
mutation_rate = 10  # Out of 100 (not decimal percentage)
initial_mutt_rate = .10
mutation_select = 'random'
parent_select = 'tournament'
crossover_select = 'single_point'
num_generations = 5000
consecutive_generations_to_stop = 1000
num_copies = 20  # This is for idea '1'
num_copies_each_type = int(num_copies/2)  # And this is for idea '2'
