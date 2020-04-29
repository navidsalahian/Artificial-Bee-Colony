import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def cost(vector):
    x = np.sum(vector * vector)
    return x


problem = {
    'var_size': 5,
    'var_max': 10,
    'var_min': 0,
    'cost_func': cost
}


def calc_prob(pop):
    mean_costs = np.mean([bee['cost'] for bee in pop])
    fitnesses = [np.exp(- bee['cost'] / mean_costs) for bee in pop]
    sum_fitness = sum(fitnesses)
    prob = [fitness / sum_fitness for fitness in fitnesses]
    return prob


def roulette_wheel(size, prob):
    indexes = np.array([i for i in range(size)])
    i = np.random.choice(indexes, 1, p=prob)[0]
    return i


def main(problem, max_iter=200, pop_size=100):
    var_size = problem['var_size']
    var_max = problem['var_max']
    var_min = problem['var_min']
    cost_func = problem['cost_func']
    limit = 0.6 * pop_size * var_size
    empty_bee = {
        'cost': None,
        'pos': None,
        'limit': 0
    }
    best_sols = []
    best_sol = {
        'cost': np.inf,
        'pos': np.zeros(var_size)
    }
    pop = []
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        empty_bee = empty_bee.copy()
        empty_bee['pos'] = np.random.uniform(var_min, var_max, var_size)
        empty_bee['cost'] = cost_func(empty_bee['pos'])
        pop.append(empty_bee)

    for it in range(max_iter):

        # employee bee phase

        for i in range(pop_size):
            j = np.random.randint(0, var_size)
            k = np.random.choice(np.concatenate((np.arange(0, i), np.arange(i+1, pop_size))), 1)[0]
            phi = np.random.uniform(-1, 1, var_size)
            new_pos = pop[i]['pos'][j].copy() + phi * (pop[i]['pos'][j].copy() - pop[k]['pos'][j].copy())
            new_cost = cost_func(new_pos)
            if new_cost < pop[i]['cost']:
                pop[i]['pos'] = new_pos
                pop[i]['cost'] = new_cost

            else:
                pop[i]['limit'] += 1

        prob = calc_prob(pop)

        # onlooker bee phase
        for m in range(pop_size):
            i = roulette_wheel(pop_size, prob)
            j = np.random.randint(0, var_size)
            k = np.random.choice(np.concatenate((np.arange(0, i), np.arange(i+1, pop_size))), 1)[0]
            phi = np.random.uniform(-1, 1, var_size)
            new_pos = pop[i]['pos'][j].copy() + phi * (pop[i]['pos'][j].copy() - pop[k]['pos'][j].copy())
            new_cost = cost_func(new_pos)
            if new_cost < pop[i]['cost']:
                pop[i]['pos'] = new_pos
                pop[i]['cost'] = new_cost

            else:
                pop[i]['limit'] += 1

        # scout bee phase
        for i in range(pop_size):
            if pop[i]['limit'] >= limit:
                pop[i]['pos'] = np.random.uniform(var_min, var_max, var_size)
                pop[i]['cos'] = cost_func(pop[i]['pos'])

        # find best sol
        for i in range(pop_size):
            if pop[i]['cost'] < best_sol['cost']:
                ff = i
                best_sol['cost'] = pop[i]['cost'].copy()
                best_sol['pos'] = pop[i]['pos'].copy()
        print("iteration {0} best cost is {1}".format(it, best_sol['cost']))
        best_sols.append(best_sol['cost'])
    print("---------------------|| Finish ||---------------------")
    print("best pos is {0}".format(best_sol['pos']))
    return best_sols

best_sols = main(problem)
font = FontProperties()
font.set_size('larger')
labels = ["Best Cost Function", "Mean Cost Function"]
plt.figure(figsize=(12.5, 4))
plt.plot(range(len(best_sols)), best_sols, label=labels[0])
plt.xlabel("Iteration #")
plt.yticks(np.arange(-5, 250, 30))
plt.ylabel("Value [-]")
plt.legend(loc="best", prop = font)
plt.grid()
plt.yscale("log")
plt.show()