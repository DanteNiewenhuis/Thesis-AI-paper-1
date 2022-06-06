# Making Hard(er) Bechmark Test Functions

This is the codebase used for the paper "Making Hard(er) Bechmark Test Functions".
In this paper we evolve default benchmark function based on their difficulty.

The codebase consists of four folders. We will now discuss the most important folders.

## project

The project folder is the most important folder that holds all code for the benchmarks,
and the algorithms that manipulate them. The Benchmarks_base folder holds the base
Benchmark class used by all benchmarks. The benchmark class is divided into four
separate files for the sake of workability. Next to the base benchmark class is
the Benchmark_2d folder. This holds the six benchmark classes we used in this work.

The Algorithms folder holds the two algorithms that were used to test benchmark hardness and evolve the benchmarks. Finally, we have the file measurements.py. This file holds all code to evaluate benchmark difficulty.

## results

The results folder consists of the results from all runs done in this work, and some images created from it. Every benchmark has a folder holding all runs in it. For all benchmarks we have only provided the run that was used for our paper.

For each run "run.csv" is created. This is a csv that provides information of each iteration in the run. Using this csv we could create the other provided files. "runs-unique.csv" is similar to "run.csv", but only shows the iterations on which the benchmark function was changed. The "frames" folder shows the different versions of the benchmark function during the run. The "annotated-frames" are similar to "frames",but have been annotated with the MOD.

## examples

The examples folder provides two examples on how to use the code. "evolve.py" allows the user to evolve one of the six benchmarks again. "show_benchmark.py" allows the user to create an interactive 3d plot of any frame in any of the runs.