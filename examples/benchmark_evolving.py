from project.Benchmarks_2d.Branin import Branin
from project.measurements import *
from project.Algorithms.FunctionEvolver import FunctionEvolver

h = FunctionEvolver(Branin, mean_objective_deficiency, base=True)
h.evolve(2000)
