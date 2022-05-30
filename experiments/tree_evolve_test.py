# %%

from project.Benchmarks_Tree.Tree import Tree
from __project_path import _
from project.Algorithms.TreeEvolver import TreeEvolver

from project.measurements import mean_objective_deficiency, get_results
# %%

evolver = TreeEvolver("x + y", mean_objective_deficiency)
# %%

evolver.step()
# %%

evolver.tree.visualize()

# %%

evolver.values
# %%


t = Tree("x+y")

# %%

bench = t.create_benchmark()
# %%
bench.plot_benchmark()
# %%
get_results(bench)
# %%
