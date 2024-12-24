# This file has fucntions to generate plot for single run experiments
# To plot results for FE, SE, Aplha updates etc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # ticks at int values

def divide_by_dim(exp_dict_SE, dim):
    for i in range(len(exp_dict_SE)):
        if exp_dict_SE[i] != None:
            exp_dict_SE[i] = exp_dict_SE[i]/dim
    return exp_dict_SE

def pegValues(exp_dict_SE):
    for i in range(len(exp_dict_SE)):
        if exp_dict_SE[i] != None and exp_dict_SE[i] > 20:
            print(f"value {exp_dict_SE[i]} pegged to 20 in iteration {i}")
            exp_dict_SE[i] = 20
        
        if exp_dict_SE[i] != None and exp_dict_SE[i] < 0:
            print(f"value {exp_dict_SE[i]} pegged to 0 in iteration {i}")
            exp_dict_SE[i] = 0

    return exp_dict_SE


# Plot SE and FE where SE is red and FE is blue
def plot_Exploration(exp_dict, iters, dim):
    # Specify figure size
    plt.figure(figsize=(12, 3))

    peg_at= 20 # worst basin fitness
    
    x= np.arange(0, iters)

    SE= pegValues(divide_by_dim(exp_dict["SE"], dim)) #exp_dict["SE"]
    FE= pegValues(divide_by_dim(exp_dict["FE"], dim)) # exp_dict["FE"]

    # Replace None with 0 (or any other value you prefer)
    #SE = [0 if v is None else v for v in exp_dict_all["SE"]]
    #FE = [0 if v is None else v for v in exp_dict_all["FE"]]


    plt.scatter(x, SE, color= 'r', label= "SE", marker= '.')
    plt.scatter(x, FE, color= 'b', label= "FE", marker= '.')

    plt.xlabel("Iterations")
    plt.ylabel("Relative Fitness")
    plt.title("Exploration Measure")

    # Ensure y-axis ticks are integers but not all integers
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # set axis to fixed size
    plt.xlim(0, len(SE))
    plt.ylim(0, peg_at) 

    # Custom y-axis ticks # easier to visualize for rastrigin int basin fitness
    #y_ticks = np.arange(0, peg_at + 1, 2)  # Modify the range and step as needed
    #plt.yticks(y_ticks)

    plt.legend()
    plt.show()


# Plot for measuring Aplha SE Distance Moved (All alpha update)
def plot_all_alpha_dist(alpha_SE):

    alpha_SE_count, alpha_all_dist, _, _, _ =alpha_SE

    alpha_dist_better, alpha_dist_worse= alpha_all_dist # unpack the values for better and worse alpha updates

    # plot alpha_SE_dist where x is the iter and y is the distance moved
    alpha_iter= np.arange(0, len(alpha_dist_better))
    
    # alpha_SE_iter is the x-axis
    # alpha_SE_dist is the y-axis
    plt.figure(figsize=(12, 3))
    plt.scatter(alpha_iter, alpha_dist_better, color= 'r', label= "Better fitness", marker= '.') # better fitness of alpha
    plt.scatter(alpha_iter, alpha_dist_worse, color= 'b', label= "Worse fitness", marker= '.')


    plt.xlabel("Iterations")
    plt.ylabel("Absolute Distance Moved")
    plt.title("Alpha Dist Measure")

    plt.xlim(0, len(alpha_dist_better))
    plt.ylim(0, )

    plt.show()


# Plot for measuring Aplha SE Distance Moved (Only SEs)
def plot_SE_alpha_dist(alpha_SE):

    alpha_SE_count, _, alpha_SE_dist, _, _ =alpha_SE

    # plot alpha_SE_dist where x is the iter and y is the distance moved
    alpha_SE_iter= np.arange(0, len(alpha_SE_dist))
    
    # alpha_SE_iter is the x-axis
    # alpha_SE_dist is the y-axis
    plt.figure(figsize=(12, 3))
    plt.scatter(alpha_SE_iter, alpha_SE_dist, color= 'r', label= "SE of Alpha", marker= '.')


    plt.xlabel("Iterations")
    plt.ylabel("Absolute Distance Moved")
    plt.title("Alpha SE Dist Measure")

    plt.xlim(0, len(alpha_SE_dist))
    plt.ylim(0, ) # np.ceil(max(alpha_SE_dist))+1

    plt.show()


# Plot for measuring Aplha Fitness 
def plot_alpha_fitness(alpha_SE):

    _, _, _, alpha_fitness_dict, _ =alpha_SE
    
    # Keys are the x-axis
    # Values are the y-axis
    plt.figure(figsize=(12, 3))
    plt.scatter(list(alpha_fitness_dict.keys()), list(alpha_fitness_dict.values()), color= 'r', label= "Alpha Fitness", marker= '.')

    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("Alpha Fitness")

    # Ensure y-axis ticks are integers but not all integers
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlim(0, len(alpha_fitness_dict))
    plt.ylim(0, np.ceil(max(list(alpha_fitness_dict.values())))+1)

    # Custom y-axis ticks # easier to visualize for rastrigin int basin fitness
    #y_ticks = np.arange(0, max(list(alpha_fitness_dict.values())) + 1, 2) 
    #plt.yticks(y_ticks)

    plt.show()


# Plot for measuring fitness of Local Basin of Reference (Alpha) solution 
def plot_L_ref_fitness(alpha_SE):

    _, _, _, _, L_ref_fitness =alpha_SE
    
    # Keys are the x-axis
    # Values are the y-axis
    plt.figure(figsize=(12, 3))
    x_axis= np.arange(0, len(L_ref_fitness))
    plt.scatter(x_axis, L_ref_fitness, color= 'g', label= "L_ref Fitness", marker= '.')

    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("L_ref Fitness: Local basin fitness of reference alpha solution")

    # Custom y-axis ticks # easier to visualize for rastrigin int basin fitness
    #y_ticks = np.arange(0, max(L_ref_fitness) + 1, 2) 
    #plt.yticks(y_ticks)

    # Ensure y-axis ticks are integers but not all integers
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlim(0, len(L_ref_fitness))
    plt.ylim(0, np.ceil(max(L_ref_fitness))+1)
    
    plt.show()