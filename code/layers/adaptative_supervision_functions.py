from math import log

from pdb import set_trace as pause

def get_adaptative_lambda(inner_iter, total_iterations):
    lb = 100
    
    lambda_gt  = (log(inner_iter+lb)-log(lb))/(log(total_iterations+lb)-log(lb))/2
    lambda_ign = 0.51 - lambda_gt
    return lambda_gt, lambda_ign 