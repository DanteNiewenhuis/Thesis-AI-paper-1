# %%

import os
import numpy as np
import numpy.typing as npt

from tqdm import tqdm

from project.measurements import get_objective, mean_objective_deficiency
from datetime import datetime
import time
from dataclasses import dataclass

from project.Benchmarks_base.benchmark import Benchmark
# %%


@dataclass
class FunctionEvolver():
    """
    Class used to evolve a given Benchmark Class. 
    The only required variable is the benchmark that will be evolved

    To evolve a benchmark the following steps are taken:
        First, create a FunctionEvolver object with the benchmark as variable.
        Second, execute FunctionEvolver.evolve()
    """
    benchmark: Benchmark

    lower: float = None
    upper: float = None
    sampling: bool = True
    base: bool = False
    D: int = 2

    ############################################################################
    # Properties
    ############################################################################

    @property
    def parameters(self) -> npt.NDArray[float]:
        return self._parameters

    @parameters.setter
    def parameters(self, _parameters: npt.NDArray[float]):
        """_summary_

        Args:
            _parameters (npt.NDArray[float]): _description_
        """
        self._parameters = _parameters

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, _value: float):
        self._value = _value

    ############################################################################
    # Base functions
    ############################################################################

    def __post_init__(self):
        """
        Function executed after the initial variables are set. 
        This function creates the initial parameters, value and logging.
        """
        # Set initial parameters
        self.init_parameters()

        # Set initial Value
        self.value = self.determine_value(self.parameters)

        self.create_logging()
        self.log_state()

    def init_parameters(self):
        """
        Initialize the parameters. 

        First, the allowed range of each benchmark is determined. 
        This is done by determening in which power of 10 the parameter falls. 

        Next, initial parameters are chosen based on the range.
        """

        # get the default parameters
        b: Benchmark = self.benchmark(D=self.D)
        params = b.params

        # determine bounds and ranges
        self.num_parameters = len(params)
        if self.lower == None or self.upper == None:
            p = np.array(params)
            p[np.where(p == 0)] = 1
            self.upper = 10 ** (
                np.floor(np.log(np.abs(p)) / np.log(10)) + 1)
            self.lower = -self.upper

        self.ranges = self.upper - self.lower

        # set initial parameters
        if self.base:
            self.parameters = np.array(params, dtype=np.float)
            return

        self.parameters = np.random.uniform(
            low=self.lower, high=self.upper, size=(self.num_parameters))

    def determine_value(self, parameters: npt.NDArray[float]) -> float:
        """Determine the value of the given set of parameters.

        Args:
            parameters (npt.NDArray[float]): parameters to use

        Returns:
            float: The value of the given parameters
        """

        return get_objective(self.benchmark(parameters, D=self.D), mean_objective_deficiency, sampling=self.sampling)

    ############################################################################
    # Logging
    ############################################################################

    def create_logging(self):
        """
        Create initial logging files.
        """
        self.logging_dir = f"results/{self.benchmark.__name__}/{datetime.today().strftime('%Y-%m-%d_%H:%M:%S')}"
        os.makedirs(self.logging_dir)

        with open(f"{self.logging_dir}/info.csv", "w") as wf:
            wf.write(f"num_params,")
            for i in range(self.num_parameters):
                wf.write(f"p_{i}_lower,p_{i}_upper,")

            wf.write(f"objective_function,dimensions,sampling,base\n")
            wf.write(f"{self.num_parameters},")

            for i in range(self.num_parameters):
                wf.write(f"{self.lower[i]},{self.upper[i]},")

            wf.write(
                f"MOD,{self.D},{self.sampling},{self.base}\n")

        self.log_file = f"{self.logging_dir}/run.csv"
        with open(self.log_file, "w") as wf:
            wf.write(f"val,")
            for i in range(self.num_parameters):
                wf.write(f"p_{i}")
                if i < (self.num_parameters-1):
                    wf.write(",")

            wf.write("\n")

    def log_state(self):
        """Log the current state of the algorithm

        Args:
            time (float): execution time of the previous step
        """
        with open(self.log_file, "a") as wf:
            wf.write(f"{self.value},")
            for i, p in enumerate(self.parameters):
                wf.write(f"{p}")
                if i < (self.num_parameters - 1):
                    wf.write(",")

            wf.write("\n")

    def step(self):
        """
            Take a step to evolve the parameters. 
            First, change one of the parameters.
            Second, determine the value of the new parameters.
            Finally, keep the new parameters if their value is higher.
        """

        # Pick which parameter to change
        dim = np.random.randint(0, self.num_parameters)
        new_params = np.copy(self.parameters)

        # change parameter
        new_params[dim] = new_params[dim] + \
            np.random.uniform(low=-0.5, high=0.5) * self.ranges[dim]

        # Bound the value
        new_params[dim] = self.upper[dim] if new_params[dim] > self.upper[dim] \
            else new_params[dim]
        new_params[dim] = self.lower[dim] if new_params[dim] < self.lower[dim] \
            else new_params[dim]

        # determine the new value
        new_value = self.determine_value(new_params)

        if (new_value > self.value):
            self.parameters = new_params
            self.value = new_value

    def evolve(self, iterations: int):
        """
        Evolve the parameters for a given number of steps. 

        Args:
            iterations (int): iterations to take
        """

        pbar = tqdm(range(iterations))
        pbar.set_description("Evolving Benchmark")
        pbar.set_postfix_str(
            f"value: {self.value:.5f}")

        for _ in pbar:
            self.step()
            pbar.set_postfix_str(
                f"value: {self.value:.5f}")

            self.log_state()
