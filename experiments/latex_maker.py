# %%

from email import header
from __project_path import _

import re
import scipy
from collections import Counter
import os
import pandas as pd

from project.Benchmarks_base.benchmark import Benchmark

from project.Benchmarks_2d.Branin import Branin
from project.Benchmarks_2d.Easom import Easom
from project.Benchmarks_2d.GoldsteinPrice import GoldsteinPrice
from project.Benchmarks_2d.MartinGaddy import MartinGaddy
from project.Benchmarks_2d.Mishra4 import Mishra4
from project.Benchmarks_2d.SixHump import SixHump
from project.measurements import *

from datetime import datetime

# %%


def latex_all_evaluations(benchmark: Benchmark, session: str):
    df = pd.read_csv(f"results/{benchmark.__name__}/{session}/run.csv")
    param_columns = [c for c in df.columns if c[:2] == "p_"]

    table = """
\\renewcommand{\\arraystretch}{2}
\\begin{table}[H]
\centering
\\resizebox{\columnwidth}{!}{
\\begin{tabular}{|r|r||r|l|}
\hline
\\textbf{Start} & \\textbf{End} & \\textbf{MOD} & \\textbf{Function} \\\\ \hline \n
"""

    for i, (val, series) in enumerate(df.groupby("val")):
        # iteration_ranges.append([series.index[0], series.index[-1]])

        params = series.iloc[0][param_columns].to_numpy()
        b: Benchmark = benchmark(params)
        table += f"{series.index[0]} & {series.index[-1]} & {val:.2e} & {b.to_latex(line_breaks=False)} \\\\ \hline \n"

    table += """
\hline
\end{tabular}
}"""
    table += f"\caption{{The evolution of the {benchmark.__name__} function. Start and End show the iteration on which they were first, and last used.}} \n"
    table += "\end{table}\\renewcommand{\\arraystretch}{1}"

    print(table)

# %%


benchmark = SixHump
sessions = [datetime.strptime(x, '%Y-%m-%d_%H:%M:%S')
            for x in os.listdir(f"results/{benchmark.__name__}")]

last_session = datetime.strftime(max(sessions), '%Y-%m-%d_%H:%M:%S')
latex_all_evaluations(benchmark, last_session)


# %%

def latex_first_last(benchmarks: list[Benchmark]):
    table = """
    \\renewcommand{\\arraystretch}{2}
\\begin{table}[H]
\centering
\\resizebox{\columnwidth}{!}{
\\begin{tabular}{|l||l|l|}
\hline
\\textbf{Benchmark} & \\textbf{Default}  & \\textbf{Evolved} \\\\ \hline \n"""

    for benchmark in benchmarks:
        sessions = [datetime.strptime(x, '%Y-%m-%d_%H:%M:%S')
                    for x in os.listdir(f"../results/{benchmark.__name__}")]

        last_session = datetime.strftime(max(sessions), '%Y-%m-%d_%H:%M:%S')

        df = pd.read_csv(
            f"../results/{benchmark.__name__}/{last_session}/run.csv")
        param_columns = [c for c in df.columns if c[:2] == "p_"]

        b_default = benchmark(df.iloc[0][param_columns].to_numpy())
        b_evolved = benchmark(df.iloc[-1][param_columns].to_numpy())

        table += f"{benchmark.__name__} & {b_default.to_latex(r=3, line_breaks=True)} & {b_evolved.to_latex(r=3, line_breaks=True)} \\\\ \hline \n"

    table += """
\hline
\end{tabular}
}\caption{Parameterized benchmark formulas before and after evolving}
\label{tab:formula_evolutions}
\end{table}
\\renewcommand{\\arraystretch}{1}"""

    print(table)


latex_first_last([Branin, Easom, GoldsteinPrice,
                 MartinGaddy, Mishra4, SixHump])
# %%


def latex_results(benchmarks: list[Benchmark]):

    table = """
\\begin{table}[H]
\centering
\\resizebox{0.8\columnwidth}{!}{
\\begin{tabular}{|l||rr||rr|}
\hline
\multirow{2}{*}{\\textbf{Benchmark}} & \multicolumn{2}{l||}{\\textbf{Mean Objective-Deficiency}} & \multicolumn{2}{l|}{\\textbf{Failure Rate}} \\\\ \cline{2-5} 
                                    & \multicolumn{1}{r|}{Default}                & Evolved      & \multicolumn{1}{r|}{Default}     & Evolved    \\\\ \hline\hline \n"""

    for benchmark in benchmarks:
        sessions = [datetime.strptime(x, '%Y-%m-%d_%H:%M:%S')
                    for x in os.listdir(f"../results/{benchmark.__name__}")]

        last_session = datetime.strftime(max(sessions), '%Y-%m-%d_%H:%M:%S')

        df = pd.read_csv(
            f"../results/{benchmark.__name__}/{last_session}/run.csv")

        param_columns = [c for c in df.columns if c[:2] == "p_"]

        b_default = benchmark(df.iloc[0][param_columns].to_numpy())
        b_evolved = benchmark(df.iloc[-1][param_columns].to_numpy())

        succ_default, _ = get_results(b_default)
        succ_evolved, _ = get_results(b_evolved)

        table += f"{benchmark.__name__} & \\multicolumn{'{1}{r|}{'}{df.iloc[0]['val']:.3e}{'}'} & {df.iloc[-1]['val']:.3e} & \\multicolumn{'{1}{r|}{'}{succ_default}\\%{'}'} & {succ_evolved}\\% \\\\ \\hline \n"

    table += """
\end{tabular}
}
\caption{Difficulty measurements for both the base, and the evolved configurations of all 2d benchmarks.}
\label{tab:results}

\end{table}
    """
    print(table)


latex_results([Branin, Easom, GoldsteinPrice, MartinGaddy, Mishra4, SixHump])


# %%


def get_df(benchmark: Benchmark, session: str = "latest"):
    if session == "latest":
        sessions = [datetime.strptime(x, '%Y-%m-%d_%H:%M:%S')
                    for x in os.listdir(f"../results/{benchmark.__name__}")]

        session = datetime.strftime(max(sessions), '%Y-%m-%d_%H:%M:%S')

    df = pd.read_csv(
        f"../results/{benchmark.__name__}/{session}/run.csv")

    df['iteration'] = df.index

    return pd.DataFrame([g[1].iloc[0] for g in df.groupby('val')])


def evolution_information(benchmark):
    df = get_df(benchmark)

    param_columns = [c for c in df.columns if "p_" in c]

    table = """
    \\renewcommand{\\arraystretch}{2}
    \\begin{table}[H]
    \centering
    \\resizebox{\columnwidth}{!}{
    \\begin{tabular}{|r||r|r|r||r|r|r|r|}
    \hline
    \\textbf{Model} & \\textbf{MOD} & \\textbf{MOD increase} & \\textbf{\%} &
                \\textbf{changed param} & \\textbf{old value} & \\textbf{new value} & \\textbf{change} \\\\ \hline 
    """

    table += f"{0} & {df.iloc[0]['val']:.3e} & & & & & & \\\\\hline \n"
    for i in range(1, len(df)):
        previous = df.iloc[i-1]
        current = df.iloc[i]

        diff = current - previous

        diff_per = diff['val'] / previous['val'] * 100

        # print(f"MOD {diff['val']}")

        diff_params = diff[param_columns]
        # print(diff_params)

        changed = diff_params[diff_params != 0]
        # print(changed)
        if len(changed) == 0:
            table += f"{i} & {current['val']:.3e} & {diff['val']:.3e} & {diff_per:.3e} & & & & \\\\\hline \n"
            continue

        changed_param = changed.index[0]
        changed_amount = changed[0]

        table += f"{i} & {current['val']:.3e} & {diff['val']:.3e} & {diff_per:.3e} & p_{'{'}{changed_param[2:]}{'}'} & {previous[changed_param]:.3f} & {current[changed_param]:.3f} & {changed_amount:.3f} \\\\\hline \n"

    table += """
    \hline
    \end{tabular}
    }\caption{"""
    table += f"The evolution of the {benchmark.__name__} function. This table shows the difference in terms of MOD and which parameter was changed in what way."
    table += """
    } 
    \end{table}\\renewcommand{\\arraystretch}{1}"""

    print(table)


evolution_information(SixHump)
# %%

for benchmark in [Branin, Easom, GoldsteinPrice, MartinGaddy, Mishra4, SixHump]:

    b: Benchmark = benchmark()
    print(b.to_latex(line_breaks=True, parameterized=True))
    print()
# %%

b = Branin()
print(b.to_latex(line_breaks=True, parameterized=True))
# %%
