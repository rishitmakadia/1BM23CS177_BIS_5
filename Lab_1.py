import random
import numpy as np
# pip install numpy

# 1. Define the Problem
def fitness_function(x):
    return x**2

# 2. Initialize Parameters
POPULATION_SIZE = 4
GENOME_LENGTH = 5  #bits
# 1 = Every pair of parents will definitely crossover.
CROSSOVER_RATE = 0.5
# 1 = Every bit of every child genome will flip.
MUTATION_RATE = 1
GENERATIONS = 3

# Decode to integer 
def decode(genome):
    return int(genome, 2)

# random 10110 type once
def generate_genome():
    return ''.join(random.choice('01') for _ in range(GENOME_LENGTH))

# random 10110 type for entire population_size
def create_population():
    return [generate_genome() for _ in range(POPULATION_SIZE)]

def evaluate(population):
    # Evaluate fitness on decoded integer (x) evaluates fitness function
    return [fitness_function(decode(ind)) for ind in population]

# selects two individuals (parents) based on their fitness probabilities
def select(population, fitnesses):
    total_fit = sum(fitnesses)
    probs = [f / total_fit for f in fitnesses]
    return random.choices(population, weights=probs, k=2)

# Combines two parent genomes (p1 and p2) to create two offspring by exchanging genetic material at a random crossover point.
def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENOME_LENGTH - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    else:
        return p1, p2

def mutate(genome):
    # Flip exactly 1 bit at a random position
    pos = random.randint(0, GENOME_LENGTH - 1)
    bit_list = list(genome)
    bit_list[pos] = '1' if bit_list[pos] == '0' else '0'
    return ''.join(bit_list)

def genetic_algorithm():
    population = create_population()

    # Show the best individual before any generation or mutation
    fitnesses = evaluate(population)
    best_initial = max(zip(population, fitnesses), key=lambda x: x[1])
    best_initial_genome, best_initial_fit = best_initial
    best_initial_x = decode(best_initial_genome)
    print("\n=== Best Solution Before Any Generation ===")
    print(f"Genome: {best_initial_genome}")
    print(f"Decoded Integer: {best_initial_x}")
    print(f"Fitness (x^2): {best_initial_fit:.4f}\n")

    for generation in range(1, GENERATIONS + 1):
        fitnesses = evaluate(population)

        # Sort population by fitness descending
        pop_fit = list(zip(population, fitnesses))
        pop_fit.sort(key=lambda x: x[1], reverse=True)

        # Calculate avg and max fitness
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)

        print(f"\n--- Generation {generation} ---")
        print(f"Average Fitness: {avg_fitness:.4f}")
        print(f"Max Fitness: {max_fitness:.4f}")

        # Show top 2 individuals BEFORE mutation
        print("Top 2 individuals BEFORE mutation:")
        for i, (genome, fit) in enumerate(pop_fit[:2], 1):
            decoded = decode(genome)
            print(f"  Rank {i}: Genome={genome} | Decoded Integer={decoded} | Fitness={fit:.4f}")

        new_population = []

        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)

            # Mutate children with exactly 1 bit flip
            mutated_child1 = mutate(child1)
            mutated_child2 = mutate(child2)

            new_population.extend([mutated_child1, mutated_child2])

        population = new_population[:POPULATION_SIZE]

        # Evaluate new population after mutation
        new_fitnesses = evaluate(population)

        # Sort new population by fitness descending
        new_pop_fit = list(zip(population, new_fitnesses))
        new_pop_fit.sort(key=lambda x: x[1], reverse=True)

        # Show top 2 individuals AFTER mutation
        print("Top 2 individuals AFTER mutation:")
        for i, (genome, fit) in enumerate(new_pop_fit[:2], 1):
            decoded = decode(genome)
            print(f"  Rank {i}: Genome={genome} | Decoded Integer={decoded} | Fitness={fit:.4f}")

    # After all generations, output overall best
    overall_best = max(new_pop_fit, key=lambda x: x[1])
    best_genome, best_fit = overall_best
    best_x = decode(best_genome)
    print("\n=== Best Solution Found After All Generations ===")
    print(f"Genome: {best_genome}")
    print(f"Decoded Integer: {best_x}")
    print(f"Fitness (x^2): {best_fit}")

genetic_algorithm()

