import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
hidden_layer_options = [(64,), (128,), (256,), (128, 64), (256, 128)]
activation_options = ["relu", "tanh"]


def random_individual():
    return {"hidden_layer_sizes": random.choice(hidden_layer_options), "activation": random.choice(activation_options),
            "alpha": 10 ** random.uniform(-5, -1), "learning_rate_init": 10 ** random.uniform(-4, -2)}


def build_pipeline(individual):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("nn", MLPClassifier(
            hidden_layer_sizes=individual["hidden_layer_sizes"],
            activation=individual["activation"],
            alpha=individual["alpha"],
            learning_rate_init=individual["learning_rate_init"],
            early_stopping=True,
            max_iter=200,
            random_state=SEED,
        )),
    ])


def evaluate_individual(individual, X, y, cv, scoring):
    pipeline = build_pipeline(individual)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores.mean()


def crossover(parent_a, parent_b):
    result = {}
    for key in parent_a:
        if random.random() < 0.5:
            result[key] = parent_a[key]
        else:
            result[key] = parent_b[key]
    
    return result
        

def mutate(individual, mutation_rate=0.3):
    mutated = dict(individual)
    if random.random() < mutation_rate:
        mutated["hidden_layer_sizes"] = random.choice(hidden_layer_options)
    if random.random() < mutation_rate:
        mutated["activation"] = random.choice(activation_options)
    if random.random() < mutation_rate:
        mutated["alpha"] = 10 ** random.uniform(-5, -1)
    if random.random() < mutation_rate:
        mutated["learning_rate_init"] = 10 ** random.uniform(-4, -2)

    return mutated


def tournament_select(population, fitnesses, k=3):
    contenders = random.sample(list(zip(population, fitnesses)), k)
    return max(contenders, key=lambda pair: pair[1])[0]


def genetic_algorithm_search(X, y, scoring, population_size, generations, cv_splits):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    population = []
    for _ in range(population_size):
        population.append(random_individual())

    best_individual = None
    best_fitness = -np.inf
    for _ in range(generations):
        fitnesses = []
        for ind in population:
            fitnesses.append(evaluate_individual(ind, X, y, cv, scoring))

        for ind, fit in zip(population, fitnesses):
            if fit > best_fitness:
                best_individual, best_fitness = ind, fit

        next_population = [best_individual]
        while len(next_population) < population_size:
            parent_a = tournament_select(population, fitnesses)
            parent_b = tournament_select(population, fitnesses)
            child = mutate(crossover(parent_a, parent_b))
            next_population.append(child)

        population = next_population

    return best_individual, best_fitness
