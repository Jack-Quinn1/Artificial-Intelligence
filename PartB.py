import random
import matplotlib.pyplot as plt

def evaluate(assignment, preferences):
    total_score = 0
    for i, l in enumerate(assignment):
        total_score += preferences[i][l]
    return total_score


def create_assignment(n_students, capacities, preferences):
    assignment = [-1] * n_students
    capacities_copy = capacities.copy()
    for i in range(n_students):
        available_lecturers = [j for j in range(len(capacities)) if capacities_copy[j] > 0]
        if not available_lecturers:
            break
        l = random.choice(available_lecturers)
        capacities_copy[l] -= 1
        assignment[i] = l
    return assignment


def generate_initial_population(preferences, capacities, size):
    n_students = len(preferences)
    return [create_assignment(n_students, capacities, preferences) for i in range(size)]


def selection(population, preferences, k):
    scores = [evaluate(assignment, preferences) for assignment in population]
    return [population[i] for i in sorted(range(len(scores)), key=lambda i: scores[i])[:k]]


def crossover(assignment1, assignment2):
    n_students = len(assignment1)
    c = random.randint(0, n_students - 1)
    return assignment1[:c] + assignment2[c:]


def mutate(assignment, preferences, capacities):
    n_students = len(assignment)
    capacities_copy = capacities.copy()
    for i in range(n_students):
        m = random.randint(0, n_students - 1)
        l = random.randint(0, len(preferences[0]) - 1)
        if capacities_copy[l] > 0:
            capacities_copy[assignment[m]] += 1
            assignment[m] = l
            capacities_copy[l] -= 1
    return assignment


def genetic_algorithm(preferences, capacities, population_size=100, k=10, num_generations=100):
    n_students = len(preferences)
    population = generate_initial_population(preferences, capacities, population_size)
    best_assignment = population[0]
    best_score = evaluate(best_assignment, preferences)
    best_scores = [best_score]

    for i in range(num_generations):
        selected_population = selection(population, preferences, k)
        new_population = []
        while len(new_population) < population_size:
            assignment1 = random.choice(selected_population)
            assignment2 = random.choice(selected_population)
            child = crossover(assignment1, assignment2)
            child = mutate(child, preferences, capacities.copy())
            new_population.append(child)
        population = new_population
        best_assignment_in_population = min(population, key=lambda x: evaluate(x, preferences))
        best_score_in_population = evaluate(best_assignment_in_population, preferences)
        if best_score_in_population < best_score:
            best_assignment = best_assignment_in_population
            best_score = best_score_in_population
        best_scores.append(best_score)
    return best_scores

def final_average_preference(preferences, assignments):
    n_students = len(preferences)
    total_preference = 0
    for i in range(n_students):
        assigned_lecturer = assignments[i]
        preference_for_lecturer = preferences[i][assigned_lecturer]
        total_preference += preference_for_lecturer
    final_average = total_preference / n_students
    return final_average

# Example usage:
n_students = 46
n_lecturers = 22
capacities = [3, 1, 2, 3, 1, 2, 2, 2, 1, 4, 1, 1, 2, 2, 3, 1, 2, 1, 1, 4, 4, 3]
preferences = [[17, 8, 9, 7, 14, 6, 5, 10, 21, 18, 20, 2, 11, 12, 4, 13, 3, 19, 1, 16, 15, 22],
               [18, 9, 7, 8, 10, 2, 1, 3, 12, 11, 13, 5, 19, 6, 20, 21, 17, 15, 22, 16, 4, 14],
               [19, 16, 10, 14, 5, 6, 3, 2, 22, 4, 17, 11, 13, 15, 7, 9, 12, 18, 20, 8, 1, 21],
               [15, 19, 21, 11, 8, 12, 2, 9, 22, 4, 7, 18, 14, 13, 17, 1, 5, 20, 3, 6, 10, 16],
               [14, 17, 2, 4, 13, 19, 20, 11, 16, 18, 21, 1, 3, 10, 6, 7, 5, 15, 12, 8, 22, 9],
               [19, 7, 9, 1, 17, 16, 2, 3, 4, 20, 21, 14, 13, 12, 6, 18, 5, 10, 8, 11, 15, 22],
               [7, 13, 4, 8, 22, 9, 3, 1, 19, 16, 21, 10, 11, 14, 18, 2, 12, 17, 5, 15, 6, 20],
               [7, 13, 2, 8, 22, 9, 3, 1, 20, 16, 18, 10, 11, 14, 19, 4, 12, 17, 5, 15, 6, 21],
               [21, 6, 11, 14, 2, 19, 1, 5, 7, 16, 22, 15, 8, 4, 13, 12, 9, 17, 3, 20, 18, 10],
               [9, 8, 14, 10, 5, 11, 3, 4, 21, 7, 13, 12, 20, 15, 17, 1, 16, 2, 6, 18, 19, 22],
               [18, 17, 16, 13, 7, 21, 1, 14, 9, 3, 20, 12, 22, 4, 19, 10, 6, 2, 8, 15, 5, 11],
               [4, 9, 7, 8, 10, 6, 11, 13, 5, 2, 1, 12, 14, 15, 16, 20, 19, 18, 21, 17, 3, 22],
               [16, 12, 11, 17, 14, 18, 1, 7, 13, 5, 15, 8, 22, 2, 21, 19, 6, 3, 9, 20, 4, 10],
               [10, 13, 9, 17, 20, 18, 2, 12, 11, 16, 17, 4, 1, 22, 5, 8, 7, 21, 6, 14, 15, 3],
               [19, 11, 9, 10, 7, 22, 3, 4, 6, 12, 21, 1, 13, 5, 16, 15, 14, 8, 20, 17, 18, 2],
               [22, 1, 2, 5, 21, 20, 10, 6, 14, 8, 19, 7, 9, 11, 4, 12, 15, 18, 3, 16, 17, 13],
               [10, 21, 20, 1, 2, 9, 3, 11, 22, 12, 13, 14, 4, 7, 19, 5, 15, 17, 16, 6, 8, 18],
               [13, 11, 12, 14, 15, 16, 1, 17, 4, 18, 19, 9, 3, 7, 8, 5, 6, 20, 21, 10, 22, 2],
               [10, 3, 4, 15, 16, 12, 9, 11, 20, 22, 17, 6, 1, 13, 21, 18, 7, 8, 5, 14, 19, 2],
               [15, 9, 11, 16, 2, 3, 4, 7, 22, 18, 19, 10, 8, 5, 20, 6, 1, 14, 12, 17, 13, 21],
               [12, 14, 15, 13, 5, 18, 1, 17, 6, 20, 21, 7, 10, 2, 16, 19, 9, 4, 11, 8, 22, 3],
               [9, 10, 11, 1, 12, 7, 4, 13, 22, 6, 14, 15, 8, 3, 16, 17, 5, 18, 19, 2, 20, 21],
               [11, 10, 9, 8, 13, 18, 12, 4, 17, 19, 22, 1, 14, 20, 7, 16, 2, 6, 5, 15, 21, 3],
               [17, 13, 10, 16, 12, 20, 3, 21, 5, 1, 4, 22, 18, 7, 9, 14, 19, 15, 8, 11, 6, 2],
               [9, 14, 13, 12, 15, 16, 4, 6, 1, 21, 18, 11, 7, 19, 8, 5, 3, 20, 2, 17, 22, 10],
               [20, 11, 12, 14, 19, 9, 3, 13, 4, 22, 10, 2, 15, 5, 16, 8, 1, 21, 7, 18, 17, 6],
               [20, 1, 6, 10, 7, 21, 12, 3, 5, 13, 22, 8, 9, 15, 2, 4, 11, 14, 18, 17, 16, 19],
               [13, 4, 3, 18, 17, 14, 11, 20, 8, 21, 15, 7, 6, 2, 9, 16, 12, 1, 10, 19, 22, 5],
               [17, 6, 2, 22, 7, 18, 4, 5, 10, 16, 21, 1, 15, 14, 12, 13, 8, 11, 3, 19, 20, 9],
               [21, 5, 18, 4, 15, 16, 1, 3, 19, 14, 13, 7, 20, 10, 6, 2, 12, 9, 17, 11, 8, 22],
               [9, 8, 2, 18, 6, 16, 17, 7, 20, 19, 21, 5, 13, 12, 10, 11, 4, 15, 3, 14, 22, 1],
               [15, 12, 11, 16, 3, 2, 1, 4, 20, 14, 22, 6, 7, 8, 17, 5, 9, 10, 19, 13, 18, 21],
               [16, 19, 18, 17, 9, 4, 2, 5, 22, 15, 20, 3, 12, 13, 14, 6, 11, 10, 7, 8, 1, 21],
               [6, 21, 11, 10, 1, 22, 20, 19, 14, 17, 12, 4, 7, 3, 13, 15, 16, 8, 9, 5, 18, 2],
               [19, 11, 9, 10, 6, 22, 1, 8, 7, 12, 21, 2, 13, 4, 16, 15, 14, 5, 20, 17, 18, 3],
               [11, 3, 8, 13, 17, 2, 1, 12, 22, 18, 19, 6, 16, 10, 9, 21, 5, 7, 4, 14, 15, 20],
               [11, 16, 21, 12, 4, 10, 5, 1, 22, 7, 20, 8, 6, 13, 17, 2, 3, 14, 19, 9, 18, 15],
               [19, 20, 16, 11, 12, 4, 1, 15, 6, 21, 14, 5, 17, 8, 18, 10, 9, 3, 22, 2, 13, 7],
               [20, 2, 17, 4, 14, 6, 1, 3, 22, 16, 19, 10, 15, 7, 8, 9, 13, 11, 12, 18, 5, 21],
               [5, 12, 4, 17, 6, 21, 7, 8, 13, 18, 19, 20, 9, 16, 10, 15, 3, 2, 11, 14, 22, 1],
               [19, 1, 20, 4, 16, 8, 2, 3, 21, 11, 18, 5, 14, 10, 17, 9, 6, 13, 12, 15, 7, 22],
               [7, 8, 11, 12, 13, 14, 5, 15, 2, 16, 17, 6, 9, 18, 19, 20, 3, 4, 10, 21, 22, 1],
               [4, 15, 17, 14, 7, 13, 16, 19, 8, 20, 18, 5, 2, 12, 3, 22, 6, 9, 10, 11, 21, 1],
               [11, 1, 2, 20, 12, 13, 5, 4, 14, 22, 21, 15, 16, 10, 6, 17, 7, 8, 3, 18, 19, 9],
               [16, 4, 5, 6, 11, 14, 7, 8, 2, 17, 15, 18, 19, 20, 3, 10, 1, 22, 12, 13, 21, 9],
               [2, 10, 3, 16, 11, 19, 9, 4, 22, 17, 18, 5, 1, 14, 20, 7, 13, 6, 8, 15, 12, 21]]
best_solution = genetic_algorithm(preferences, capacities)
print("Best solution:", best_solution)
# final_avg_preference = final_average_preference(preferences, best_solution)
# print("Final average preference:", final_avg_preference)

best_scores = genetic_algorithm(preferences, capacities)
plt.plot(best_scores)
plt.xlabel("Generation")
plt.ylabel("Best Score")
plt.show()