# %%

import plotly.graph_objects as go

import numpy as np
from dataclasses import dataclass

from typing import Tuple
number = int | float | complex


# %%


@dataclass
class PPA():
    benchmark: list  # This is Benchmark
    pop_size: int = 30
    n_max: int = 5

    ############################################################################
    # Properties
    ############################################################################

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, _population):
        self._population = _population

    ############################################################################
    # basic class functionality
    ############################################################################

    def __post_init__(self):
        self.evaluations = 0

        self.lower_bounds = self.benchmark.lower_bounds
        self.upper_bounds = self.benchmark.upper_bounds
        self.D = len(self.lower_bounds)
        self.feature_range = self.upper_bounds - self.lower_bounds

        self.init_pop()

    def init_pop(self):
        """ Create an initial population of random values between the bounds
        """
        self.populations = []
        self.population = np.random.uniform(
            low=self.lower_bounds, high=self.upper_bounds, size=(self.pop_size, self.D))

    def calculate_fitness(self, population: list[list[number]]) -> Tuple[list[number], list[number]]:
        """Calculate the fitness of the elements in the population and normalize them

        Args:
            population (list): A population of sprouts

        Returns:
            Tuple[list[number], list[number]]: The abstract and the normalized 
                objective value of the given population
        """
        f = self.benchmark.get_value(population)
        self.evaluations += len(population)
        if (f.max() == f.min()):
            z = np.ones_like(f) * 0.5
        else:
            z = (f - f.min()) / (f.max() - f.min())

        F = 0.5*(np.tanh(4*z-2) + 1)

        return f, F

    def calc_population_fitness(self):
        f, F = self.calculate_fitness(self.population)

        self.f = f
        self.F = F

        return f, F

    def get_current_fitness(self) -> Tuple[number, number, number]:
        """Return the best performing sprout in the current population

        Returns:
            Tuple[number, number, number]: [x, y, z] of the best performing sprout
        """
        f, F = self.calc_population_fitness()

        i_min = np.argmin(f)

        return np.append(self.population[i_min, :], f[i_min])

    ############################################################################
    # Evolution functionality
    ############################################################################

    def get_offspring(self, F: list[number]) -> list[list[number]]:
        """Get the offspring of the current population, based on the fitness, and 
        return them together with the current population.

        Args:
            F (list[number]): The normalized fitness of the current population

        Returns:
            list[list[number]]: new population consisting of the current population, 
                and all offspring
        """
        num_off = np.ceil(
            self.n_max * F * np.random.uniform(low=0, high=1)).astype(int)

        mutation = 1 - F

        offspring = np.array([])
        for p, n, m in zip(self.population, num_off, mutation):
            c = 2 * np.random.uniform(low=-0.5, high=0.5,
                                      size=(n, self.D))*m*self.feature_range
            p_new = p+c

            if len(offspring) == 0:
                offspring = p_new
            else:
                offspring = np.concatenate(
                    (offspring, p_new), axis=0)

        # Bounds correction
        for i, (lower, higher) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            offspring[np.where(offspring[:, i] < lower), i] = lower
            offspring[np.where(offspring[:, i] > higher), i] = higher

        f, _ = self.calculate_fitness(offspring)
        return offspring, f

    def select_pop(self, offspring: list[list[number]], f: list[list[number]]) -> list[list[number]]:
        """Select a new population from the offspring by picking the popSize top 
        performing sprouts

        Args:
            offspring (list[list[number]]): The offspring of the current population

        Returns:
            list[list[number]]: The best performing sprouts of the offspring
        """

        args = np.argsort(f)[:self.pop_size]

        return offspring[args], f[args]

    def next_generation(self, f, F) -> Tuple[list[number], list[number]]:
        """Determine the next generation.

        Returns:
            Tuple[list[number], list[number]]: The abstract and the normalized 
                objective value of the new population 
        """

        offspring, f_off = self.get_offspring(F)

        f = np.concatenate((f, f_off))

        offspring = np.concatenate(
            (self.population, offspring))

        new_pop, f = self.select_pop(offspring, f)
        self.population = new_pop

        return f, F

    def evolve(self, max_iterations: int):
        """Evolve the population for g_max iterations

        Args:
            g_max (int): Number of iterations to evolve
        Returns:
            list[number]: The fitness of the model at each iteration
        """

        self.init_pop()
        f, F = self.calc_population_fitness()
        while self.evaluations < max_iterations:
            f, F = self.next_generation(f, F)
