# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
from pathlib import Path
import optimizers.PSO as pso
import optimizers.MVO as mvo
import optimizers.GWO as gwo
import optimizers.GWOM as gwom
import optimizers.MFO as mfo
import optimizers.CS as cs
import optimizers.BAT as bat
import optimizers.WOA as woa
import optimizers.FFA as ffa
import optimizers.SSA as ssa
import optimizers.GA as ga
import optimizers.HHO as hho
import optimizers.SCA as sca
import optimizers.JAYA as jaya
import optimizers.DE as de
import benchmarks
import csv
import numpy
import time
import warnings
import os
import plot_convergence as conv_plot
import plot_boxplot as box_plot
import shutil

warnings.simplefilter(action="ignore")


def selector(algo, func_details, popSize, Iter, no_repeat):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]

    if algo == "SSA":
        x = ssa.SSA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "PSO":
        x = pso.PSO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "GA":
        x = ga.GA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "BAT":
        x = bat.BAT(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "FFA":
        x = ffa.FFA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "GWO":
        x = gwo.GWO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "GWOM":
        x = gwom.GWOM(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "WOA":
        x = woa.WOA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "MVO":
        x = mvo.MVO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "MFO":
        x = mfo.MFO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "CS":
        x = cs.CS(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "HHO":
        x = hho.HHO(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "SCA":
        x = sca.SCA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "JAYA":
        x = jaya.JAYA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    elif algo == "DE":
        x = de.DE(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, no_repeat)
    else:
        return null
    return x


def run(optimizer, objectivefunc, NumOfRuns, params, export_flags, dim, out_dir):

    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    optimizer : list
        The list of optimizers names
    objectivefunc : list
        The list of benchmark functions
    NumOfRuns : int
        The number of independent runs
    params  : set
        The set of parameters which are:
        1. Size of population (PopulationSize)
        2. The number of iterations (Iterations)
    export_flags : set
        The set of Boolean flags which are:
        1. Export (Exporting the results in a file)
        2. Export_details (Exporting the detailed results in files)
        3. Export_convergence (Exporting the covergence plots)
        4. Export_boxplot (Exporting the box plots)
    dim : int 
        The number of dimensions defining the complexity of the problem

    Returns
    -----------
    N/A
    """

    # Select general parameters for all optimizers (population size, number of iterations) ....
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]
    
    # Check if single run or multiple runs
    if NumOfRuns == 1:
        no_repeat = True
    else:
        no_repeat= False

    # Export results ?
    Export = export_flags["Export_avg"]
    Export_details = export_flags["Export_details"]
    Export_convergence = export_flags["Export_convergence"]
    Export_boxplot = export_flags["Export_boxplot"]

    Flag = False
    Flag_details = False

    # CSV Header for for the cinvergence
    CnvgHeader = []

    # Dict to store all failed exploration results
    if no_repeat:
        failed_exp_dict= {}

    #results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    results_directory = out_dir + "/"
    results_path= Path(results_directory)
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True, exist_ok=False)

    for l in range(0, Iterations):
        CnvgHeader.append("Iter" + str(l + 1))

    for i in range(0, len(optimizer)):
        for j in range(0, len(objectivefunc)):
            convergence = [0] * NumOfRuns
            executionTime = [0] * NumOfRuns
            for k in range(0, NumOfRuns):
                func_details = benchmarks.getFunctionDetails(objectivefunc[j], dim)
                x = selector(optimizer[i], func_details, PopulationSize, Iterations, no_repeat)
                convergence[k] = x.convergence
                optimizerName = x.optimizer
                objfname = x.objfname
                if Export_details == True:
                    ExportToFile = results_directory + "experiment_details.csv"
                    with open(ExportToFile, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if (
                            Flag_details == False
                        ):  # just one time to write the header of the CSV file
                            header = numpy.concatenate(
                                [["Optimizer", "objfname", "ExecutionTime", "Individual"], CnvgHeader]
                            )
                            writer.writerow(header)
                            Flag_details = True  # at least one experiment
                        executionTime[k] = x.executionTime
                        a = numpy.array([x.optimizer, x.objfname, x.executionTime, x.bestIndividual] + x.convergence.tolist(), dtype=object) #  x.convergence.tolist() is fitness at every iter
                        writer.writerow(a)
                    out.close()
                
                # Collect data on FE if single run
                if no_repeat:
                    alpha_SE= (x.alpha_SE_count, (x.alpha_dist_better, x.alpha_dist_worse), x.SE_alpha_dist, x.alpha_fitness, x.L_ref_fitness) 
                    failed_exp_results= (x.exp_dict, alpha_SE)

                    failed_exp_dict[f"{optimizerName}_{objfname}_{k}"]= failed_exp_results

            if Export == True:
                ExportToFile = results_directory + "experiment.csv"

                with open(ExportToFile, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")
                    if (
                        Flag == False
                    ):  # just one time to write the header of the CSV file
                        header = numpy.concatenate(
                            [["Optimizer", "objfname", "ExecutionTime"], CnvgHeader]
                        )
                        writer.writerow(header)
                        Flag = True

                    avgExecutionTime = float("%0.2f" % (sum(executionTime) / NumOfRuns))
                    avgConvergence = numpy.around(
                        numpy.mean(convergence, axis=0, dtype=numpy.float64), decimals=2
                    ).tolist()
                    a = numpy.concatenate([[optimizerName, objfname, avgExecutionTime], avgConvergence])
                    writer.writerow(a)
                out.close()

    if Export_convergence == True:
        conv_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    if Export_boxplot == True:
        box_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    if Flag == False:  # Faild to run at least one experiment
        print(
            "No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions"
        )

    print("Execution completed")

    if no_repeat:
        return failed_exp_dict
    else:
        return None
