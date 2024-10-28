import network_functions as nf
import genetic_functions as gf
import parameters as p
import pygad
from numpy import array
from networkx import write_edgelist

# some global variables
same_fitness_counter = 0
best_fitness_last_gen = None


def on_generation(ga_instance):
    global same_fitness_counter, best_fitness_last_gen

    # Get the best fitness in the current generation
    best_fitness = ga_instance.best_solution()[1]

    # Check if the best fitness is the same as the last generation
    if best_fitness_last_gen is not None and best_fitness == best_fitness_last_gen:
        same_fitness_counter += 1
    else:
        same_fitness_counter = 0

    # Update the last generation's best fitness
    best_fitness_last_gen = best_fitness

    # Print current generation and best fitness (for debugging)
    print(
        f"Generation: {ga_instance.generations_completed} " +
        f"Best Fitness: {best_fitness}, Same Fitness Counter: {same_fitness_counter}")

    # Stop if the same fitness value has persisted for 1000 generations
    if same_fitness_counter >= p.consecutive_generations_to_stop:
        print("Stopping early; the fitness value has been the same for 1000 generations.")
        return 'stop'


def run_all():
    G = nf.G
    num_genes = G.number_of_edges()

    # ---------- Create Initial Gene Pool ---------- #
    gene_pool = gf.generate_initial_population(G, p.speedy_demand, '2')
    initial_population = array(gene_pool)
    print(f'Initial population: \n{initial_population}')

    # ---------- Set up PyGAD instance ---------- #
    ga_instance = pygad.GA(
        num_generations=p.num_generations,          # Number of generations to evolve
        num_parents_mating=4,                       # Number of parents for mating
        fitness_func=gf.fitness_opt,                # Fitness function
        num_genes=num_genes,                        # Number of genes per chromosome
        initial_population=initial_population,      # Initial population
        mutation_percent_genes=p.mutation_rate,     # Mutation percentage
        crossover_type=p.crossover_select,          # Crossover type
        # mutation_type='inversion',                # Mutation type
        keep_elitism=0,                             # Keep 0 prev. solutions
        gene_space=[0, 1],                          # Possible gene values
        gene_type=int,                              # Data type of gene
        on_generation=on_generation                 # Function to stop
    )

    # Running the genetic algorithm
    ga_instance.run()

    # After the algorithm completes
    ga_instance.plot_fitness()

    # Get the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    print(f"Best Solution: {solution}")
    print(f"Best Solution Fitness: {solution_fitness}")

    gf.fitness(solution, G, p.speedy_demand, True, 'Best solution')

    G_s = nf.report_results(solution, p.location)

    return G_s


def run_and_prune():
    # First run the GA to get initial solution
    G_s = run_all()
    print('Initial Genetic Run complete\nTrimming useless edges now.')

    # Prune all edges not included in the shortest path for any demand pair
    G_s2 = nf.prune_dead_branches(G_s, p.speedy_demand)

    if input('Would you like to save this graph? y/n') == 'y':
        write_edgelist(G_s2, 'best_solution_graph')

    if input('Test this graph for feasibility y/n') == 'y':
        nf.verify_solution(nf.G, G_s2, p.speedy_demand)
