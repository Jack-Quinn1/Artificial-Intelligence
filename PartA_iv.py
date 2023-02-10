import random
import matplotlib.pyplot as plt

def generate_population(pop_size, string_length):
    population = []
    for i in range(pop_size):
        chromosome = [random.randint(0, 9) for j in range(string_length)]
        population.append(chromosome)
    return population

def calculate_fitness(chromosome, target):
    count = sum([1 for i in range(len(chromosome)) if chromosome[i] == target[i]])
    return count

def select_parents(population, fitness_vals):
    parent1, parent2 = random.choices(population, weights=fitness_vals, k=2)
    return parent1, parent2

def crossover(parent1, parent2, crossover_point):
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(chromosome, mutation_prob):
    for i in range(len(chromosome)):
        if random.uniform(0, 1) < mutation_prob:
            chromosome[i] = random.randint(0, 9)
    return chromosome

def evolve(population, fitness_vals, pop_size, mutation_prob):
    new_population = []
    for i in range(pop_size):
        parent1, parent2 = select_parents(population, fitness_vals)
        crossover_point = random.randint(0, len(parent1))
        child = crossover(parent1, parent2, crossover_point)
        child = mutate(child, mutation_prob)
        new_population.append(child)
    return new_population

def plot_fitness(fitness_history):
    plt.plot(fitness_history)
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness")
    plt.show()

if __name__ == "__main__":
    string_length = 30
    pop_size = 1000
    num_generations = 100
    mutation_prob = 0.01
    target = [random.randint(0, 9) for i in range(string_length)]
    target_fitness = string_length

    population = generate_population(pop_size, string_length)
    fitness_history = []
    for generation in range(num_generations):
        fitness_vals = [calculate_fitness(chromosome, target) for chromosome in population]
        avg_fitness = sum(fitness_vals) / pop_size
        fitness_history.append(avg_fitness)
        if avg_fitness == target_fitness:
            break
        population = evolve(population, fitness_vals, pop_size, mutation_prob)
    plot_fitness(fitness_history)
