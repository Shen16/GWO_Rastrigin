# -*- coding: utf-8 -*-
"""
Created on Mon May 16 00:27:50 2016
@author: Stephen Chen
GWOM: Grey Wolf Optimizer Modified
"""

import random
import numpy
import math
from solution import solution
import time
from failed_exploration import measure_exploration, closest_integer_minima
import gc


def GWOM(objf, lb, ub, dim, SearchAgents_no, Max_iter, no_repeat):
    gc.collect()

    # Max_iter=1000
    # lb=-100
    # ub=100
    # dim=30
    # SearchAgents_no=5

    cycles= 5
    # Specify upper and lower bounds of random numbers
    r_lb= 0.25
    r_up= 0.75

    # initialize alpha, beta, and delta_pos
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = numpy.zeros(Max_iter) # fitness of solution (alpha) at every iter
    s = solution()

    # initialize datapoints to collect for failed exploration
    if no_repeat:
        s.SE_alpha_dist= numpy.zeros(Max_iter*SearchAgents_no)
        s.alpha_dist_better= numpy.full(Max_iter*SearchAgents_no, numpy.nan)
        s.alpha_dist_worse= numpy.full(Max_iter*SearchAgents_no, numpy.nan)

        # Need to specify size of aplha dicts as well (since not every iter alpha gets updated)
        s.exp_dict["SE"]= [None]*(Max_iter*SearchAgents_no) 
        s.exp_dict["DE"]= [None]*(Max_iter*SearchAgents_no)
        s.exp_dict["FE"]= [None]*(Max_iter*SearchAgents_no)
        s.exp_dict["SR"]= [None]*(Max_iter*SearchAgents_no)

    # Loop counter
    print('GWO is optimizing  "' + objf.__name__ + '"') 

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    # Main loop
    iter_par_i =0
    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])  # omega? Yes # All omegas computed at once in the bottom # search solutions

            # Store old alpha to collect FE info
            old_Alpha_pos= Alpha_pos.copy()

            # -------  MEASURE EXPLORATION (should be done before alpha update)-----------
            # Alpha only updates under this if statement so collect info on failed exp here # collect only if repeats=1
            if no_repeat: # If single run collect details on Failed Exp
                search_pos= Positions[i, :].copy()
                ref_pos = Alpha_pos.copy()
                return_list1 = measure_exploration(ref_pos, search_pos, objf)            
                # if SE, FE, DE, or SR (when local attraction basin is different) collect data
                if return_list1[0] != "None":
                    exp_type, relF_curr = return_list1[0], return_list1[1]
                    for key in s.exp_dict:
                        if key== exp_type:
                            s.exp_dict[key][iter_par_i]= relF_curr
                            #print("UPDATED LIST:", s.exp_dict[key][iter_par_i])
                            if return_list1[0]== "SE":
                                print("SUCESSFUL EXPLORATION")
                
                # -------  Track fitness of local basin of ref soln (only for rastrigin)----------
                if objf.__name__ in ["F9", "F24"]: # rastrigin or shifted rastrigin
                    s.L_ref_fitness.append(objf(closest_integer_minima(ref_pos)))

                # ----------  Track fitness of alpha ------------
                s.alpha_fitness[iter_par_i]= objf(Alpha_pos)
                    
            # ------------    END  -------------------

            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score: # Alpha gets updated so SE or DE
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                # Update alpha
                Alpha_pos = Positions[i, :].copy()
                

            if fitness > Alpha_score and fitness < Beta_score:
                Delta_score = Beta_score  # Update delte
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness  # Update beta
                Beta_pos = Positions[i, :].copy()

            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness  # Update delta
                Delta_pos = Positions[i, :].copy()
            
            # -------  MEASUREMENTS FOR ALPHA UPDATES (should be done after alpha update)-----------
            
            # --------- Measure dist moved for all alpha updates ---------
            ref_dist_moved= abs(numpy.linalg.norm(numpy.array(old_Alpha_pos) - numpy.array(Alpha_pos))) # measure all dist of Alpha updates
            if objf(Alpha_pos)< objf(old_Alpha_pos): #better
                s.alpha_dist_better[iter_par_i]= ref_dist_moved
            else:
                s.alpha_dist_worse[iter_par_i] = ref_dist_moved

            # ------- Measure SE of Alpha  -----------
            return_list2= measure_exploration(old_Alpha_pos, Alpha_pos, objf)
            if return_list2[0] != "None": # if not exploitation
                exp_type, relF_curr = return_list2[0], return_list2[1]
                if exp_type== "SE":
                    s.alpha_SE_count+= 1
                    # measure euclidean dist between old and new alpha
                    distance= abs(numpy.linalg.norm(numpy.array(old_Alpha_pos) -numpy.array(Alpha_pos))) # L2 Norm # taking abs to measure magnitude only
                    s.SE_alpha_dist[iter_par_i]= distance
                else:
                    s.SE_alpha_dist[iter_par_i]= None # exploration but not SE
            else:
                s.SE_alpha_dist[iter_par_i]= None # exploitation
                    
            # ------------    END  -------------------
            
            iter_par_i +=1
            gc.collect()

        a = 2 - l * ((2) / Max_iter)
        # a decreases linearly from 2 to 0

        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                
                # Alternate between exploration and exploitation every 5 iterations
                if (l // cycles) % 2 == 0:
                    # Exploration: r1 < 0.25 or r1 > 0.75
                    r1_a = random.choice([random.uniform(0, r_lb), random.uniform(r_up, 1)])
                    r1_b = random.choice([random.uniform(0, r_lb), random.uniform(r_up, 1)])
                    r1_d = random.choice([random.uniform(0, r_lb), random.uniform(r_up, 1)])
                else:
                    # Exploitation: 0.25 < r1 < 0.75
                    r1_a = random.uniform(r_lb, r_up)
                    r1_b = random.uniform(r_lb, r_up)
                    r1_d = random.uniform(r_lb, r_up)
                
                r2_a = random.random()  # r2 is a random number in [0,1]
                r2_b = random.random() 
                r2_d = random.random() 

                A1 = 2 * a * r1_a - a  # -a to a # |A|>1 (diverge from prey when random values are in [-1,1]), |A|<1 (converge towards prey)
                # Equation (3.3)
                C1 = 2 * r2_a # 0 to 2 # c>1 (emphasize effect of prey in defining dist) vs c<1 (de-emphasize effect of prey) 
                # Equation (3.4)

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                # Equation (3.5)-part 1
                X1 = Alpha_pos[j] - A1 * D_alpha
                # Equation (3.6)-part 1

                A2 = 2 * a * r1_b - a
                # Equation (3.3)
                C2 = 2 * r2_b
                # Equation (3.4)

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 2
                X2 = Beta_pos[j] - A2 * D_beta
                # Equation (3.6)-part 2

                A3 = 2 * a * r1_d - a
                # Equation (3.3)
                C3 = 2 * r2_d
                # Equation (3.4)

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                # Equation (3.5)-part 3
                X3 = Delta_pos[j] - A3 * D_delta
                # Equation (3.5)-part 3

                Positions[i, j] = (X1 + X2 + X3) / 3  # Equation (3.7) # updates position of omega wolf in particular din
            
        Convergence_curve[l] = Alpha_score

        if l % 1 == 0:
            print(["At iteration " + str(l) + " the best fitness is " + str(Alpha_score)])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve  
    s.optimizer = "GWO"
    s.bestIndividual = Alpha_pos
    s.objfname = objf.__name__

    return s
