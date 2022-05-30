# %%

from project.Benchmarks_Tree.Tree import CustomBenchmark, Tree

from project.measurements import get_results, get_objective, mean_objective_deficiency

import numpy as np

# %%
tree = Tree("x^2  + y^2")

# %%

tree.visualize()

# %%
bench = tree.create_benchmark()

bench.plot_benchmark()

# %%
###################################################################
# Changing
###################################################################
tree = Tree("x + 5 * y")
tree.visualize()

# %%
tree.change_random_node()
print(tree)
tree.visualize()

# %%
###################################################################
# Adition
###################################################################
tree = Tree("x + 5 * y")
tree.visualize()
# %%
tree.add_random_node()
print(tree)
tree.visualize()

# %%

# %%
###################################################################
# Adition
###################################################################
tree = Tree("x + 5 * y + 7 * 3")
tree.visualize()

# %%
tree.trim_random_node()
print(tree)
tree.visualize()
