import random
import numpy
import gc
# ----- Rastrigin only

# Rounding to the nearest integer minima
def closest_integer_minima(position):
    return numpy.array([round(x) for x in position])


def measure_exploration(ref_pos, search_pos, objfunc): # position vectors
    gc.collect()
    expType= "None"
    return_list=[] 

    # Compute fitness
    F_ref = objfunc(ref_pos) # alpha/beta/gamma 
    F_search = objfunc(search_pos) #serach 

    # Get local attraction basin
    L_ref = closest_integer_minima(ref_pos)
    L_search = closest_integer_minima(search_pos)

    # Compute fitness of local attraction basin
    LF_ref = objfunc(L_ref)
    LF_search = objfunc(L_search)

    # Compute relative fitness of old (current) pos from it's local attraction basin
    relF_search= F_search - LF_search  # relative fitness of search (new) soln

    # Compute exploration measure
    #print("L_ref", L_ref.tolist())
    #print("L_search", L_search.tolist())
    if (L_ref.tolist() != L_search.tolist()): # comparing basin position vectors (not fitness as diff basins may have same fitness)
        # Exploration
        
        # Compare fitness of pos and local basin
        # better basin
        if LF_search < LF_ref:
            if F_search < F_ref: # better soln
                expType= "SE" # successful exploration
            else:
                expType= "FE" # failed exploration
        # worse basin
        else: 
            if F_search < F_ref: # better soln
                expType= "DE" # deceptive exploration
            else: 
                expType= "SR" # successful rejection
    
    return_list.append(expType)
    return_list.append(relF_search)   
    return return_list
