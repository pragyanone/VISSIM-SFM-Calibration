"""
VISSIM SFM Parameter Calibration using Genetic Algorithm
-------------------------------------------------------------

This script calibrates the Social Force Model (pedestrian walking behaviour) parameters in VISSIM using a genetic algorithm (GA) optimization approach.
The goal is to find parameter values that minimize the difference between simulated and observed pedestrian speeds.

Author: github.com/pragyanone
Based on: github/navs-svan/VISSIM-Pedestrian-Calibration
"""

import pygad
import shutil
import os
import win32com.client as com
import numpy as np
import pandas as pd


def set_vissim(resolution, seed, filename):
    """Initialize and configure a Vissim simulation instance.

    Args:
        resolution (int): Simulation resolution in simulation steps per second
        seed (int): Random seed for the simulation
        filename (str): Path to the Vissim network file (.inpx)

    Returns:
        com.Dispatch: Configured Vissim COM object
    """
    Vissim = com.Dispatch("Vissim.Vissim")
    Vissim.LoadNet(os.path.join(os.getcwd(), filename))

    End_of_simulation = 3600
    Vissim.Simulation.SetAttValue("SimPeriod", End_of_simulation)
    Vissim.Simulation.SetAttValue("SimRes", resolution)
    Vissim.Simulation.SetAttValue("RandSeed", seed)

    Vissim.Evaluation.SetAttValue("PedNetPerfCollectData", 1)
    Vissim.Evaluation.SetAttValue("PedRecWriteFile", 1)

    Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)
    Vissim.Simulation.SetAttValue("UseMaxSimSpeed", True)
    Vissim.SuspendUpdateGUI()

    for simRun in Vissim.Net.SimulationRuns:
        Vissim.Net.SimulationRuns.RemoveSimulationRun(simRun)
    return Vissim


def set_parameters(Vissim, parameter_list):
    """Set walking behavior parameters in Vissim simulation.

    Args:
        Vissim (com.Dispatch): Vissim COM object
        parameter_list (list): List of 7 parameter values in order:
            [Tau, ASocIso, BSocIso, Lambda, ASocMean, BSocMean, VD]
    """
    parameter_names = [
        "Tau",
        "ASocIso",
        "BSocIso",
        "Lambda",
        "ASocMean",
        "BSocMean",
        "VD",
    ]
    parameter_dict = {
        parameter_names[i]: parameter_list[i] for i in range(len(parameter_names))
    }
    Vissim.Simulation.Stop()
    for key, value in parameter_dict.items():
        Vissim.Net.WalkingBehaviors.ItemByKey(1).SetAttValue(key, value)


def get_data(SimCounter, filename):
    """Process Vissim pedestrian output data and calculate speed statistics.

    Args:
        SimCounter (int): Simulation counter for file naming
        filename (str): Base filename for input/output files

    Returns:
        list: Sorted list of average pedestrian speeds
    """
    input_file = f"{filename}_{SimCounter:03}.pp"
    output_csv = f"gotdata_{filename}_{SimCounter:03}.csv"
    analysis_csv = f"pedestrian_speed_{filename}_{SimCounter:03}.csv"

    with open(input_file, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.startswith("$PEDESTRIAN"):
            headers = line.strip().replace("$PEDESTRIAN:", "").split(";")
            data_lines = lines[i + 1 :]
            break
    else:
        raise ValueError("$PEDESTRIAN section not found in file")

    data = [line.strip().split(";") for line in data_lines if line.strip()]
    df = pd.DataFrame(data, columns=headers)

    df["NO"] = df["NO"].astype(int)
    df["SPEED"] = df["SPEED"].astype(float)
    df["CONSTRELNO"] = df["CONSTRELNO"].astype(int)

    df.to_csv(output_csv, index=False)

    df_filtered = df[
        (df["SPEED"] >= 0.2)
        & (df["CONSTRELTYPE"] == "Pedestrian link")
        & df["CONSTRELNO"].isin([1, 2])
    ]

    df_avg = df_filtered.groupby("NO")[["SPEED"]].mean().reset_index()
    df_avg.to_csv(analysis_csv, index=False)

    df["PEDROUTSTA\\NO"] = df["PEDROUTSTA\\NO"].astype(int)
    flow_counts = df.groupby("PEDROUTSTA\\NO")["NO"].nunique()
    with open("flow_check.txt", "a") as f:
        f.write(
            f"{filename}_{SimCounter:03}, {flow_counts.get(1, 0)}, {flow_counts.get(2, 0)}\n"
        )

    return sorted(df_avg["SPEED"].tolist())


def len_equalizer(list1, list2):
    """Equalize lengths of two lists by trimming the longer one from both ends.

    Args:
        list1 (list): First input list
        list2 (list): Second input list

    Returns:
        tuple: Two lists of equal length
    """
    diff = len(list1) - len(list2)
    if diff == 0:
        return list1, list2
    longer, shorter = (list1, list2) if diff > 0 else (list2, list1)
    trim = abs(diff) // 2
    longer = longer[trim : trim + len(shorter)]
    return (longer, shorter) if diff > 0 else (shorter, longer)


def rmspe(actual, simulated):
    """Calculate Root Mean Square Percentage Error between two datasets.

    Args:
        actual (array-like): Observed/actual values
        simulated (array-like): Simulated values

    Returns:
        float: RMSPE value
    """
    actual, simulated = len_equalizer(np.sort(actual), np.sort(simulated))
    return np.sqrt(np.mean(((actual - simulated) / (actual + 1e-10)) ** 2))


def read_frequency_table(filename):
    """Read CSV file containing speed frequency distribution data.

    Args:
        filename (str): Path to CSV file with 'speed' and 'no' columns

    Returns:
        numpy.ndarray: Expanded array of speed values
    """
    df = pd.read_csv(filename)
    data = np.repeat(df["speed"].values, df["no"].values)
    return data[data > 0]


def read_speed_data(filename):
    """Read CSV file containing individual pedestrian speed data.

    Args:
        filename (str): Path to CSV file with 'SPEED' column

    Returns:
        numpy.ndarray: Array of speed values
    """
    df = pd.read_csv(filename)
    return df["SPEED"].values


def fitness_func(ga_instance, solution, solution_idx):
    """Fitness function for genetic algorithm optimization.

    Args:
        ga_instance (pygad.GA): Genetic algorithm instance
        solution (list): Current parameter solution
        solution_idx (int): Index of current solution

    Returns:
        float: Fitness value (inverse of RMSPE)
    """
    global SimCounter, filename
    set_parameters(Vissim, solution)
    Vissim.Simulation.RunContinuous()

    SimResult = get_data(SimCounter, filename)
    output = rmspe(actual_speeds, SimResult)

    log = f"Simulation: {SimCounter} \t {[f'{item:05.2f}' for item in solution]} \t rmspe: {output:.4f}"

    print(log)
    with open(f"output-{filename}.txt", "a") as file:
        file.write(log + "\n")

    fitness = round(1 / (output + 1e-8), 2)
    SimCounter += 1
    return fitness


def on_generation(ga_instance):
    """Callback function executed after each GA generation.

    Args:
        ga_instance (pygad.GA): Genetic algorithm instance
    """
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    log = f"\nGeneration {ga_instance.generations_completed}: Best Solution = {[f"{item:05.2f}" for item in best_solution]}, Fitness = {best_solution_fitness:.2f}, RMSPE ={1/best_solution_fitness:.4f}"
    print(log)
    with open(f"output-{filename}.txt", "a") as file:
        file.write(log + "\n")


def genetic_algorithm():
    """Run genetic algorithm optimization for pedestrian parameter calibration."""
    global SimCounter
    fitness_function = fitness_func
    num_generations = 50
    num_parents_mating = 3
    sol_per_pop = 15
    num_genes = 7
    parent_selection_type = "tournament"
    keep_elitism = 2
    crossover_type = "uniform"
    mutation_type = "random"
    mutation_percent_genes = 20
    gene_space = [
        {"low": 0.2, "high": 2, "step": 0.1},  # Tau
        {"low": 0, "high": 5, "step": 0.2},  # ASocIso
        {"low": 0.01, "high": 0.5, "step": 0.05},  # BSocIso
        {"low": 0, "high": 0.6, "step": 0.05},  # Lambda
        {"low": 0, "high": 1, "step": 0.05},  # AsocMean
        {"low": 0.01, "high": 5, "step": 0.5},  # BsocMean
        {"low": 0, "high": 5, "step": 1},  # VD
    ]

    # PARAMETER       DEFAULT VALUE
    # Tau             0.4
    # ASocIso         2.72
    # BSocIso         0.2
    # Lambda          0.176
    # ASocMean        0.4
    # BSocMean        2.8
    # VD              3

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_space=gene_space,
        parent_selection_type=parent_selection_type,
        keep_elitism=keep_elitism,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        stop_criteria=["saturate_10"],
        on_generation=on_generation,
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(
        f"\n\nParameters of the best solution : {[f'{param:.2f}' for param in solution]}"
    )
    print(f"Fitness value of the best solution = {solution_fitness:.2f}")
    print(f"rmspe value of the best solution = {1 / solution_fitness:.2f}")


SimCounter = 1
actual_speeds_file = "civil_actual.csv"
try:
    actual_speeds = read_frequency_table(actual_speeds_file)
except KeyError:
    actual_speeds = read_speed_data(actual_speeds_file)

filename = "civil"
Vissim = set_vissim(resolution=10, seed=10410, filename=filename + ".inpx")
genetic_algorithm()

for i in range(4):
    seed = np.random.randint(500, 20000)
    new_filename = f"{filename}-{seed}"
    shutil.copy(filename + ".inpx", new_filename + ".inpx")
    Vissim = set_vissim(resolution=10, seed=seed, filename=new_filename + ".inpx")
    SimCounter = 1
    filename = new_filename
    genetic_algorithm()
