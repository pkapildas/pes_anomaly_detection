# ============================================
# CONFIGURATION - Edit these values as needed
# ============================================
TRAIN_STEP = 128  # Step size for training data loader

train_configs = {
    "train_epochs": 1,
    "batch_size": 1000,
    "optim": "adam",
    "learning_rate": 1e-4,
    "lradj": "type1",
    "patience": 3
}
model_configs_0 = {
    "seq_len": 128,
    "patch_len": 4,

    "ms_kernels": [32, 16, 8, 4],
    "ms_method": "average_pooling",

    "topk": 10,
    "n_query": 5,
    "query_len": 5,
    "bank_size": 32,
    "decay": 0.95,
    "epsilon": 1e-5,

    "e_layers": 2,
    
    "d_layers": 2,
    "m_layers": 2,

    "n_heads": 4,
    "attn_dropout": 0.1,
    "proj_dropout": 0.1,

    "d_model": 128,
    "d_ff": None,
    "ff_dropout": 0.1,
    "norm": "layernorm",
    "activation": "gelu"
}

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')



#------------ accomplish -----------------------
import numpy as np
import pandas as pd

def accomplish_evaluate(results_storage, metrics, labels, score, **args):
    if "accomplish_UCR" in metrics:
        result = {}
        sorted_indexs = np.argsort(-score)        # DESC
        # find the first overlap index
        topk = 0
        for i, (index) in enumerate(sorted_indexs):
            if labels[index] == 1:
                topk = i + 1
                break
            
        result['topk'] = topk
        result['total_len'] = len(score)
        aplha_quantile = topk/len(score)
        result['aplha_quantile'] = aplha_quantile
        result['3_alpha'] = aplha_quantile < 0.03
        result['10_alpha'] = aplha_quantile < 0.1
        results_storage['accomplish_UCR'] = pd.DataFrame([result])


#-------------------- affiliation zone ------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def t_start(j, Js = [(1,2),(3,4),(5,6)], Trange = (1,10)):
    """
    Helper for `E_gt_func`
    
    :param j: index from 0 to len(Js) (included) on which to get the start
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included
    :return: generalized start such that the middle of t_start and t_stop 
    always gives the affiliation zone
    """
    b = max(Trange)
    n = len(Js)
    if j == n:
        return(2*b - t_stop(n-1, Js, Trange))
    else:
        return(Js[j][0])

def t_stop(j, Js = [(1,2),(3,4),(5,6)], Trange = (1,10)):
    """
    Helper for `E_gt_func`
    
    :param j: index from 0 to len(Js) (included) on which to get the stop
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included
    :return: generalized stop such that the middle of t_start and t_stop 
    always gives the affiliation zone
    """
    if j == -1:
        a = min(Trange)
        return(2*a - t_start(0, Js, Trange))
    else:
        return(Js[j][1])

def E_gt_func(j, Js, Trange):
    """
    Get the affiliation zone of element j of the ground truth
    
    :param j: index from 0 to len(Js) (excluded) on which to get the zone
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included, can 
    be (-math.inf, math.inf) for distance measures
    :return: affiliation zone of element j of the ground truth represented
    as a couple
    """
    range_left = (t_stop(j-1, Js, Trange) + t_start(j, Js, Trange))/2
    range_right = (t_stop(j, Js, Trange) + t_start(j+1, Js, Trange))/2
    return((range_left, range_right))

def get_all_E_gt_func(Js, Trange):
    """
    Get the affiliation partition from the ground truth point of view
    
    :param Js: ground truth events, as a list of couples
    :param Trange: range of the series where Js is included, can 
    be (-math.inf, math.inf) for distance measures
    :return: affiliation partition of the events
    """
    # E_gt is the limit of affiliation/attraction for each ground truth event
    E_gt = [E_gt_func(j, Js, Trange) for j in range(len(Js))]
    return(E_gt)

def affiliation_partition(Is = [(1,1.5),(2,5),(5,6),(8,9)], E_gt = [(1,2.5),(2.5,4.5),(4.5,10)]):
    """
    Cut the events into the affiliation zones
    The presentation given here is from the ground truth point of view,
    but it is also used in the reversed direction in the main function.
    
    :param Is: events as a list of couples
    :param E_gt: range of the affiliation zones
    :return: a list of list of intervals (each interval represented by either 
    a couple or None for empty interval). The outer list is indexed by each
    affiliation zone of `E_gt`. The inner list is indexed by the events of `Is`.
    """
    out = [None] * len(E_gt)
    for j in range(len(E_gt)):
        E_gt_j = E_gt[j]
        discarded_idx_before = [I[1] < E_gt_j[0] for I in Is]  # end point of predicted I is before the begin of E
        discarded_idx_after = [I[0] > E_gt_j[1] for I in Is] # start of predicted I is after the end of E
        kept_index = [not(a or b) for a, b in zip(discarded_idx_before, discarded_idx_after)]
        Is_j = [x for x, y in zip(Is, kept_index)]
        out[j] = [interval_intersection(I, E_gt[j]) for I in Is_j]
    return(out)


#------------------------------ integral interval --------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
"""
In order to shorten the length of the variables,
the general convention in this file is to let:
    - I for a predicted event (start, stop),
    - Is for a list of predicted events,
    - J for a ground truth event,
    - Js for a list of ground truth events.
"""

def interval_length(J = (1,2)):
    """
    Length of an interval
    
    :param J: couple representating the start and stop of an interval, or None
    :return: length of the interval, and 0 for a None interval
    """
    if J is None:
        return(0)
    return(J[1] - J[0])

def sum_interval_lengths(Is = [(1,2),(3,4),(5,6)]):
    """
    Sum of length of the intervals
    
    :param Is: list of intervals represented by starts and stops
    :return: sum of the interval length
    """
    return(sum([interval_length(I) for I in Is]))

def interval_intersection(I = (1, 3), J = (2, 4)): 
    """
    Intersection between two intervals I and J
    I and J should be either empty or represent a positive interval (no point)
    
    :param I: an interval represented by start and stop
    :param J: a second interval of the same form
    :return: an interval representing the start and stop of the intersection (or None if empty)
    """
    if I is None:
        return(None)
    if J is None:
        return(None)
        
    I_inter_J = (max(I[0], J[0]), min(I[1], J[1]))
    if I_inter_J[0] >= I_inter_J[1]:
        return(None)
    else:
        return(I_inter_J)

def interval_subset(I = (1, 3), J = (0, 6)):
    """
    Checks whether I is a subset of J
    
    :param I: an non empty interval represented by start and stop
    :param J: a second non empty interval of the same form
    :return: True if I is a subset of J
    """
    if (I[0] >= J[0]) and (I[1] <= J[1]):
        return True
    else:
        return False

def cut_into_three_func(I, J):
    """
    Cut an interval I into a partition of 3 subsets:
        the elements before J,
        the elements belonging to J,
        and the elements after J
    
    :param I: an interval represented by start and stop, or None for an empty one
    :param J: a non empty interval
    :return: a triplet of three intervals, each represented by either (start, stop) or None
    """
    if I is None:
        return((None, None, None))
    
    I_inter_J = interval_intersection(I, J)
    if I == I_inter_J:
        I_before = None
        I_after = None
    elif I[1] <= J[0]:
        I_before = I
        I_after = None
    elif I[0] >= J[1]:
        I_before = None
        I_after = I
    elif (I[0] <= J[0]) and (I[1] >= J[1]):
        I_before = (I[0], I_inter_J[0])
        I_after = (I_inter_J[1], I[1])
    elif I[0] <= J[0]:
        I_before = (I[0], I_inter_J[0])
        I_after = None
    elif I[1] >= J[1]:
        I_before = None
        I_after = (I_inter_J[1], I[1])
    else:
        raise ValueError('unexpected unconsidered case')
    return(I_before, I_inter_J, I_after)
  
def get_pivot_j(I, J):
    """
    Get the single point of J that is the closest to I, called 'pivot' here,
    with the requirement that I should be outside J
    
    :param I: a non empty interval (start, stop)
    :param J: another non empty interval, with empty intersection with I
    :return: the element j of J that is the closest to I
    """
    if interval_intersection(I, J) is not None:
        raise ValueError('I and J should have a void intersection')

    j_pivot = None # j_pivot is a border of J
    if max(I) <= min(J):
        j_pivot = min(J)
    elif min(I) >= max(J):
        j_pivot = max(J)
    else:
        raise ValueError('I should be outside J')
    return(j_pivot)

def integral_mini_interval(I, J):
    """
    In the specific case where interval I is located outside J,
    integral of distance from x to J over the interval x \in I.
    This is the *integral* i.e. the sum.
    It's not the mean (not divided by the length of I yet)
    
    :param I: a interval (start, stop), or None
    :param J: a non empty interval, with empty intersection with I
    :return: the integral of distances d(x, J) over x \in I
    """
    if I is None:
        return(0)

    j_pivot = get_pivot_j(I, J)
    a = min(I)
    b = max(I)
    return((b-a)*abs((j_pivot - (a+b)/2)))

def integral_interval_distance(I, J):
    """
    For any non empty intervals I, J, compute the
    integral of distance from x to J over the interval x \in I.
    This is the *integral* i.e. the sum. 
    It's not the mean (not divided by the length of I yet)
    The interval I can intersect J or not
    
    :param I: a interval (start, stop), or None
    :param J: a non empty interval
    :return: the integral of distances d(x, J) over x \in I
    """
    # I and J are single intervals (not generic sets)
    # I is a predicted interval in the range of affiliation of J
    
    def f(I_cut):
        return(integral_mini_interval(I_cut, J))
    # If I_middle is fully included into J, it is
    # the distance to J is always 0
    def f0(I_middle):
        return(0)

    cut_into_three = cut_into_three_func(I, J)
    # Distance for now, not the mean:
    # Distance left: Between cut_into_three[0] and the point min(J)
    d_left = f(cut_into_three[0])
    # Distance middle: Between cut_into_three[1] = I inter J, and J
    d_middle = f0(cut_into_three[1])
    # Distance right: Between cut_into_three[2] and the point max(J)
    d_right = f(cut_into_three[2])
    # It's an integral so summable
    return(d_left + d_middle + d_right)

def integral_mini_interval_P_CDFmethod__min_piece(I, J, E):
    """
    Helper of `integral_mini_interval_Pprecision_CDFmethod`
    In the specific case where interval I is located outside J,
    compute the integral $\int_{d_min}^{d_max} \min(m, x) dx$, with:
    - m the smallest distance from J to E,
    - d_min the smallest distance d(x, J) from x \in I to J
    - d_max the largest distance d(x, J) from x \in I to J
    
    :param I: a single predicted interval, a non empty interval (start, stop)
    :param J: ground truth interval, a non empty interval, with empty intersection with I
    :param E: the affiliation/influence zone for J, represented as a couple (start, stop)
    :return: the integral $\int_{d_min}^{d_max} \min(m, x) dx$
    """
    if interval_intersection(I, J) is not None:
        raise ValueError('I and J should have a void intersection')
    if not interval_subset(J, E):
        raise ValueError('J should be included in E')
    if not interval_subset(I, E):
        raise ValueError('I should be included in E')

    e_min = min(E)
    j_min = min(J)
    j_max = max(J)
    e_max = max(E)
    i_min = min(I)
    i_max = max(I)
  
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    m = min(j_min - e_min, e_max - j_max)
    A = min(d_max, m)**2 - min(d_min, m)**2
    B = max(d_max, m) - max(d_min, m)
    C = (1/2)*A + m*B
    return(C)

def integral_mini_interval_Pprecision_CDFmethod(I, J, E):
    """
    Integral of the probability of distances over the interval I.
    In the specific case where interval I is located outside J,
    compute the integral $\int_{x \in I} Fbar(dist(x,J)) dx$.
    This is the *integral* i.e. the sum (not the mean)
    
    :param I: a single predicted interval, a non empty interval (start, stop)
    :param J: ground truth interval, a non empty interval, with empty intersection with I
    :param E: the affiliation/influence zone for J, represented as a couple (start, stop)
    :return: the integral $\int_{x \in I} Fbar(dist(x,J)) dx$
    """
    integral_min_piece = integral_mini_interval_P_CDFmethod__min_piece(I, J, E)
  
    e_min = min(E)
    j_min = min(J)
    j_max = max(J)
    e_max = max(E)
    i_min = min(I)
    i_max = max(I)
    d_min = max(i_min - j_max, j_min - i_max)
    d_max = max(i_max - j_max, j_min - i_min)
    integral_linear_piece = (1/2)*(d_max**2 - d_min**2)
    integral_remaining_piece = (j_max - j_min)*(i_max - i_min)
    
    DeltaI = i_max - i_min
    DeltaE = e_max - e_min
    
    output = DeltaI - (1/DeltaE)*(integral_min_piece + integral_linear_piece + integral_remaining_piece)
    return(output)

def integral_interval_probaCDF_precision(I, J, E):
    """
    Integral of the probability of distances over the interval I.
    Compute the integral $\int_{x \in I} Fbar(dist(x,J)) dx$.
    This is the *integral* i.e. the sum (not the mean)
    
    :param I: a single (non empty) predicted interval in the zone of affiliation of J
    :param J: ground truth interval
    :param E: affiliation/influence zone for J
    :return: the integral $\int_{x \in I} Fbar(dist(x,J)) dx$
    """
    # I and J are single intervals (not generic sets)
    def f(I_cut):
        if I_cut is None:
            return(0)
        else:
            return(integral_mini_interval_Pprecision_CDFmethod(I_cut, J, E))
            
    # If I_middle is fully included into J, it is
    # integral of 1 on the interval I_middle, so it's |I_middle|
    def f0(I_middle):
        if I_middle is None:
            return(0)
        else:
            return(max(I_middle) - min(I_middle))
    
    cut_into_three = cut_into_three_func(I, J)
    # Distance for now, not the mean:
    # Distance left: Between cut_into_three[0] and the point min(J)
    d_left = f(cut_into_three[0])
    # Distance middle: Between cut_into_three[1] = I inter J, and J
    d_middle = f0(cut_into_three[1])
    # Distance right: Between cut_into_three[2] and the point max(J)
    d_right = f(cut_into_three[2])
    # It's an integral so summable
    return(d_left + d_middle + d_right)

def cut_J_based_on_mean_func(J, e_mean):
    """
    Helper function for the recall.
    Partition J into two intervals: before and after e_mean
    (e_mean represents the center element of E the zone of affiliation)
    
    :param J: ground truth interval
    :param e_mean: a float number (center value of E)
    :return: a couple partitionning J into (J_before, J_after)
    """
    if J is None:
        J_before = None
        J_after = None
    elif e_mean >= max(J):
        J_before = J
        J_after = None
    elif e_mean <= min(J):
        J_before = None
        J_after = J
    else: # e_mean is across J
        J_before = (min(J), e_mean)
        J_after = (e_mean, max(J))
        
    return((J_before, J_after))

def integral_mini_interval_Precall_CDFmethod(I, J, E):
    """
    Integral of the probability of distances over the interval J.
    In the specific case where interval J is located outside I,
    compute the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$.
    This is the *integral* i.e. the sum (not the mean)
    
    :param I: a single (non empty) predicted interval
    :param J: ground truth (non empty) interval, with empty intersection with I
    :param E: the affiliation/influence zone for J, represented as a couple (start, stop)
    :return: the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$
    """
    # The interval J should be located outside I 
    # (so it's either the left piece or the right piece w.r.t I)
    i_pivot = get_pivot_j(J, I)
    e_min = min(E)
    e_max = max(E)
    e_mean = (e_min + e_max) / 2
    
    # If i_pivot is outside E (it's possible), then
    # the distance is worst that any random element within E,
    # so we set the recall to 0
    if i_pivot <= min(E):
        return(0)
    elif i_pivot >= max(E):
        return(0)
    # Otherwise, we have at least i_pivot in E and so d < M so min(d,M)=d
    
    cut_J_based_on_e_mean = cut_J_based_on_mean_func(J, e_mean)
    J_before = cut_J_based_on_e_mean[0]
    J_after = cut_J_based_on_e_mean[1]
  
    iemin_mean = (e_min + i_pivot)/2
    cut_Jbefore_based_on_iemin_mean = cut_J_based_on_mean_func(J_before, iemin_mean)
    J_before_closeE = cut_Jbefore_based_on_iemin_mean[0] # before e_mean and closer to e_min than i_pivot ~ J_before_before
    J_before_closeI = cut_Jbefore_based_on_iemin_mean[1] # before e_mean and closer to i_pivot than e_min ~ J_before_after
    
    iemax_mean = (e_max + i_pivot)/2
    cut_Jafter_based_on_iemax_mean = cut_J_based_on_mean_func(J_after, iemax_mean)
    J_after_closeI = cut_Jafter_based_on_iemax_mean[0] # after e_mean and closer to i_pivot than e_max ~ J_after_before
    J_after_closeE = cut_Jafter_based_on_iemax_mean[1] # after e_mean and closer to e_max than i_pivot ~ J_after_after
    
    if J_before_closeE is not None:
        j_before_before_min = min(J_before_closeE) # == min(J)
        j_before_before_max = max(J_before_closeE)
    else:
        j_before_before_min = math.nan
        j_before_before_max = math.nan
  
    if J_before_closeI is not None:
        j_before_after_min = min(J_before_closeI) # == j_before_before_max if existing
        j_before_after_max = max(J_before_closeI) # == max(J_before)
    else:
        j_before_after_min = math.nan
        j_before_after_max = math.nan
   
    if J_after_closeI is not None:
        j_after_before_min = min(J_after_closeI) # == min(J_after)
        j_after_before_max = max(J_after_closeI) 
    else:
        j_after_before_min = math.nan
        j_after_before_max = math.nan
    
    if J_after_closeE is not None:
        j_after_after_min = min(J_after_closeE) # == j_after_before_max if existing
        j_after_after_max = max(J_after_closeE) # == max(J)
    else:
        j_after_after_min = math.nan
        j_after_after_max = math.nan
  
    # <-- J_before_closeE --> <-- J_before_closeI --> <-- J_after_closeI --> <-- J_after_closeE -->
    # j_bb_min       j_bb_max j_ba_min       j_ba_max j_ab_min      j_ab_max j_aa_min      j_aa_max
    # (with `b` for before and `a` for after in the previous variable names)
    
    #                                          vs e_mean  m = min(t-e_min, e_max-t)  d=|i_pivot-t|   min(d,m)                            \int min(d,m)dt   \int d dt        \int_(min(d,m)+d)dt                                    \int_{t \in J}(min(d,m)+d)dt
    # Case J_before_closeE & i_pivot after J   before     t-e_min                    i_pivot-t       min(i_pivot-t,t-e_min) = t-e_min    t^2/2-e_min*t     i_pivot*t-t^2/2  t^2/2-e_min*t+i_pivot*t-t^2/2 = (i_pivot-e_min)*t      (i_pivot-e_min)*tB - (i_pivot-e_min)*tA = (i_pivot-e_min)*(tB-tA)
    # Case J_before_closeI & i_pivot after J   before     t-e_min                    i_pivot-t       min(i_pivot-t,t-e_min) = i_pivot-t  i_pivot*t-t^2/2   i_pivot*t-t^2/2  i_pivot*t-t^2/2+i_pivot*t-t^2/2 = 2*i_pivot*t-t^2      2*i_pivot*tB-tB^2 - 2*i_pivot*tA + tA^2 = 2*i_pivot*(tB-tA) - (tB^2 - tA^2)
    # Case J_after_closeI & i_pivot after J    after      e_max-t                    i_pivot-t       min(i_pivot-t,e_max-t) = i_pivot-t  i_pivot*t-t^2/2   i_pivot*t-t^2/2  i_pivot*t-t^2/2+i_pivot*t-t^2/2 = 2*i_pivot*t-t^2      2*i_pivot*tB-tB^2 - 2*i_pivot*tA + tA^2 = 2*i_pivot*(tB-tA) - (tB^2 - tA^2)
    # Case J_after_closeE & i_pivot after J    after      e_max-t                    i_pivot-t       min(i_pivot-t,e_max-t) = e_max-t    e_max*t-t^2/2     i_pivot*t-t^2/2  e_max*t-t^2/2+i_pivot*t-t^2/2 = (e_max+i_pivot)*t-t^2  (e_max+i_pivot)*tB-tB^2 - (e_max+i_pivot)*tA + tA^2 = (e_max+i_pivot)*(tB-tA) - (tB^2 - tA^2)
    #
    # Case J_before_closeE & i_pivot before J  before     t-e_min                    t-i_pivot       min(t-i_pivot,t-e_min) = t-e_min    t^2/2-e_min*t     t^2/2-i_pivot*t  t^2/2-e_min*t+t^2/2-i_pivot*t = t^2-(e_min+i_pivot)*t  tB^2-(e_min+i_pivot)*tB - tA^2 + (e_min+i_pivot)*tA = (tB^2 - tA^2) - (e_min+i_pivot)*(tB-tA)
    # Case J_before_closeI & i_pivot before J  before     t-e_min                    t-i_pivot       min(t-i_pivot,t-e_min) = t-i_pivot  t^2/2-i_pivot*t   t^2/2-i_pivot*t  t^2/2-i_pivot*t+t^2/2-i_pivot*t = t^2-2*i_pivot*t      tB^2-2*i_pivot*tB - tA^2 + 2*i_pivot*tA = (tB^2 - tA^2) - 2*i_pivot*(tB-tA)
    # Case J_after_closeI & i_pivot before J   after      e_max-t                    t-i_pivot       min(t-i_pivot,e_max-t) = t-i_pivot  t^2/2-i_pivot*t   t^2/2-i_pivot*t  t^2/2-i_pivot*t+t^2/2-i_pivot*t = t^2-2*i_pivot*t      tB^2-2*i_pivot*tB - tA^2 + 2*i_pivot*tA = (tB^2 - tA^2) - 2*i_pivot*(tB-tA)
    # Case J_after_closeE & i_pivot before J   after      e_max-t                    t-i_pivot       min(t-i_pivot,e_max-t) = e_max-t    e_max*t-t^2/2     t^2/2-i_pivot*t  e_max*t-t^2/2+t^2/2-i_pivot*t = (e_max-i_pivot)*t      (e_max-i_pivot)*tB - (e_max-i_pivot)*tA = (e_max-i_pivot)*(tB-tA)
    
    if i_pivot >= max(J):
        part1_before_closeE = (i_pivot-e_min)*(j_before_before_max - j_before_before_min) # (i_pivot-e_min)*(tB-tA) # j_before_before_max - j_before_before_min
        part2_before_closeI = 2*i_pivot*(j_before_after_max-j_before_after_min) - (j_before_after_max**2 - j_before_after_min**2) # 2*i_pivot*(tB-tA) - (tB^2 - tA^2) # j_before_after_max - j_before_after_min
        part3_after_closeI = 2*i_pivot*(j_after_before_max-j_after_before_min) - (j_after_before_max**2 - j_after_before_min**2) # 2*i_pivot*(tB-tA) - (tB^2 - tA^2) # j_after_before_max - j_after_before_min  
        part4_after_closeE = (e_max+i_pivot)*(j_after_after_max-j_after_after_min) - (j_after_after_max**2 - j_after_after_min**2) # (e_max+i_pivot)*(tB-tA) - (tB^2 - tA^2) # j_after_after_max - j_after_after_min
        out_parts = [part1_before_closeE, part2_before_closeI, part3_after_closeI, part4_after_closeE]
    elif i_pivot <= min(J):
        part1_before_closeE = (j_before_before_max**2 - j_before_before_min**2) - (e_min+i_pivot)*(j_before_before_max-j_before_before_min) # (tB^2 - tA^2) - (e_min+i_pivot)*(tB-tA) # j_before_before_max - j_before_before_min
        part2_before_closeI = (j_before_after_max**2 - j_before_after_min**2) - 2*i_pivot*(j_before_after_max-j_before_after_min) # (tB^2 - tA^2) - 2*i_pivot*(tB-tA) # j_before_after_max - j_before_after_min
        part3_after_closeI = (j_after_before_max**2 - j_after_before_min**2) - 2*i_pivot*(j_after_before_max - j_after_before_min) # (tB^2 - tA^2) - 2*i_pivot*(tB-tA) # j_after_before_max - j_after_before_min
        part4_after_closeE = (e_max-i_pivot)*(j_after_after_max - j_after_after_min) # (e_max-i_pivot)*(tB-tA) # j_after_after_max - j_after_after_min
        out_parts = [part1_before_closeE, part2_before_closeI, part3_after_closeI, part4_after_closeE]
    else:
        raise ValueError('The i_pivot should be outside J')
    
    out_integral_min_dm_plus_d = _sum_wo_nan(out_parts) # integral on all J, i.e. sum of the disjoint parts

    # We have for each point t of J:
    # \bar{F}_{t, recall}(d) = 1 - (1/|E|) * (min(d,m) + d)
    # Since t is a single-point here, and we are in the case where i_pivot is inside E.
    # The integral is then given by:
    # C = \int_{t \in J} \bar{F}_{t, recall}(D(t)) dt
    #   = \int_{t \in J} 1 - (1/|E|) * (min(d,m) + d) dt
    #   = |J| - (1/|E|) * [\int_{t \in J} (min(d,m) + d) dt]
    #   = |J| - (1/|E|) * out_integral_min_dm_plus_d    
    DeltaJ = max(J) - min(J)
    DeltaE = max(E) - min(E)
    C = DeltaJ - (1/DeltaE) * out_integral_min_dm_plus_d
    
    return(C)

def integral_interval_probaCDF_recall(I, J, E):
    """
    Integral of the probability of distances over the interval J.
    Compute the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$.
    This is the *integral* i.e. the sum (not the mean)

    :param I: a single (non empty) predicted interval
    :param J: ground truth (non empty) interval
    :param E: the affiliation/influence zone for J
    :return: the integral $\int_{y \in J} Fbar_y(dist(y,I)) dy$
    """
    # I and J are single intervals (not generic sets)
    # E is the outside affiliation interval of J (even for recall!)
    # (in particular J \subset E)
    #
    # J is the portion of the ground truth affiliated to I
    # I is a predicted interval (can be outside E possibly since it's recall)
    def f(J_cut):
        if J_cut is None:
            return(0)
        else:
            return integral_mini_interval_Precall_CDFmethod(I, J_cut, E)

    # If J_middle is fully included into I, it is
    # integral of 1 on the interval J_middle, so it's |J_middle|
    def f0(J_middle):
        if J_middle is None:
            return(0)
        else:
            return(max(J_middle) - min(J_middle))
    
    cut_into_three = cut_into_three_func(J, I) # it's J that we cut into 3, depending on the position w.r.t I
    # since we integrate over J this time.
    #
    # Distance for now, not the mean:
    # Distance left: Between cut_into_three[0] and the point min(I)
    d_left = f(cut_into_three[0])
    # Distance middle: Between cut_into_three[1] = J inter I, and I
    d_middle = f0(cut_into_three[1])
    # Distance right: Between cut_into_three[2] and the point max(I)
    d_right = f(cut_into_three[2])
    # It's an integral so summable
    return(d_left + d_middle + d_right)


#----------------------- single ground---------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

def affiliation_precision_distance(Is = [(1,2),(3,4),(5,6)], J = (2,5.5)):
    """
    Compute the individual average distance from Is to a single ground truth J
    
    :param Is: list of predicted events within the affiliation zone of J
    :param J: couple representating the start and stop of a ground truth interval
    :return: individual average precision directed distance number
    """
    if all([I is None for I in Is]): # no prediction in the current area
        return(math.nan) # undefined
    return(sum([integral_interval_distance(I, J) for I in Is]) / sum_interval_lengths(Is))

def affiliation_precision_proba(Is = [(1,2),(3,4),(5,6)], J = (2,5.5), E = (0,8)):
    """
    Compute the individual precision probability from Is to a single ground truth J
    
    :param Is: list of predicted events within the affiliation zone of J
    :param J: couple representating the start and stop of a ground truth interval
    :param E: couple representing the start and stop of the zone of affiliation of J
    :return: individual precision probability in [0, 1], or math.nan if undefined
    """
    if all([I is None for I in Is]): # no prediction in the current area
        return(math.nan) # undefined
    return(sum([integral_interval_probaCDF_precision(I, J, E) for I in Is]) / sum_interval_lengths(Is))

def affiliation_recall_distance(Is = [(1,2),(3,4),(5,6)], J = (2,5.5)):
    """
    Compute the individual average distance from a single J to the predictions Is
    
    :param Is: list of predicted events within the affiliation zone of J
    :param J: couple representating the start and stop of a ground truth interval
    :return: individual average recall directed distance number
    """
    Is = [I for I in Is if I is not None] # filter possible None in Is
    if len(Is) == 0: # there is no prediction in the current area
        return(math.inf)
    E_gt_recall = get_all_E_gt_func(Is, (-math.inf, math.inf))  # here from the point of view of the predictions
    Js = affiliation_partition([J], E_gt_recall) # partition of J depending of proximity with Is
    return(sum([integral_interval_distance(J[0], I) for I, J in zip(Is, Js)]) / interval_length(J))

def affiliation_recall_proba(Is = [(1,2),(3,4),(5,6)], J = (2,5.5), E = (0,8)):
    """
    Compute the individual recall probability from a single ground truth J to Is
    
    :param Is: list of predicted events within the affiliation zone of J
    :param J: couple representating the start and stop of a ground truth interval
    :param E: couple representing the start and stop of the zone of affiliation of J
    :return: individual recall probability in [0, 1]
    """
    Is = [I for I in Is if I is not None] # filter possible None in Is
    if len(Is) == 0: # there is no prediction in the current area
        return(0)
    E_gt_recall = get_all_E_gt_func(Is, E) # here from the point of view of the predictions
    Js = affiliation_partition([J], E_gt_recall) # partition of J depending of proximity with Is
    return(sum([integral_interval_probaCDF_recall(I, J[0], E) for I, J in zip(Is, Js)]) / interval_length(J))


#---------------- generics ---------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import groupby
from operator import itemgetter
import math
import gzip
import glob
import os

def convert_vector_to_events(vector = [0, 1, 1, 0, 0, 1, 0]):
    """
    Convert a binary vector (indicating 1 for the anomalous instances)
    to a list of events. The events are considered as durations,
    i.e. setting 1 at index i corresponds to an anomalous interval [i, i+1).
    
    :param vector: a list of elements belonging to {0, 1}
    :return: a list of couples, each couple representing the start and stop of
    each event
    """
    positive_indexes = [idx for idx, val in enumerate(vector) if val > 0]
    events = []
    for k, g in groupby(enumerate(positive_indexes), lambda ix : ix[0] - ix[1]):
        cur_cut = list(map(itemgetter(1), g))
        events.append((cur_cut[0], cur_cut[-1]))
    
    # Consistent conversion in case of range anomalies (for indexes):
    # A positive index i is considered as the interval [i, i+1),
    # so the last index should be moved by 1
    events = [(x, y+1) for (x,y) in events]
        
    return(events)

def infer_Trange(events_pred, events_gt):
    """
    Given the list of events events_pred and events_gt, get the
    smallest possible Trange corresponding to the start and stop indexes 
    of the whole series.
    Trange will not influence the measure of distances, but will impact the
    measures of probabilities.
    
    :param events_pred: a list of couples corresponding to predicted events
    :param events_gt: a list of couples corresponding to ground truth events
    :return: a couple corresponding to the smallest range containing the events
    """
    if len(events_gt) == 0:
        raise ValueError('The gt events should contain at least one event')
    if len(events_pred) == 0:
        # empty prediction, base Trange only on events_gt (which is non empty)
        return(infer_Trange(events_gt, events_gt))
        
    min_pred = min([x[0] for x in events_pred])
    min_gt = min([x[0] for x in events_gt])
    max_pred = max([x[1] for x in events_pred])
    max_gt = max([x[1] for x in events_gt])
    Trange = (min(min_pred, min_gt), max(max_pred, max_gt))
    return(Trange)

def has_point_anomalies(events):
    """
    Checking whether events contain point anomalies, i.e.
    events starting and stopping at the same time.
    
    :param events: a list of couples corresponding to predicted events
    :return: True is the events have any point anomalies, False otherwise
    """
    if len(events) == 0:
        return(False)
    return(min([x[1] - x[0] for x in events]) == 0)

def _sum_wo_nan(vec):
    """
    Sum of elements, ignoring math.isnan ones
    
    :param vec: vector of floating numbers
    :return: sum of the elements, ignoring math.isnan ones
    """
    vec_wo_nan = [e for e in vec if not math.isnan(e)]
    return(sum(vec_wo_nan))
    
def _len_wo_nan(vec):
    """
    Count of elements, ignoring math.isnan ones
    
    :param vec: vector of floating numbers
    :return: count of the elements, ignoring math.isnan ones
    """
    vec_wo_nan = [e for e in vec if not math.isnan(e)]
    return(len(vec_wo_nan))

def read_gz_data(filename = 'data/machinetemp_groundtruth.gz'):
    """
    Load a file compressed with gz, such that each line of the
    file is either 0 (representing a normal instance) or 1 (representing)
    an anomalous instance.
    :param filename: file path to the gz compressed file
    :return: list of integers with either 0 or 1
    """
    with gzip.open(filename, 'rb') as f:
        content = f.read().splitlines()
    content = [int(x) for x in content]
    return(content)

def read_all_as_events():
    """
    Load the files contained in the folder `data/` and convert
    to events. The length of the series is kept.
    The convention for the file name is: `dataset_algorithm.gz`
    :return: two dictionaries:
        - the first containing the list of events for each dataset and algorithm,
        - the second containing the range of the series for each dataset
    """
    filepaths = glob.glob('data/*.gz')
    datasets = dict()
    Tranges = dict()
    for filepath in filepaths:
        vector = read_gz_data(filepath)
        events = convert_vector_to_events(vector)
        # ad hoc cut for those files
        cut_filepath = (os.path.split(filepath)[1]).split('_')
        data_name = cut_filepath[0]
        algo_name = (cut_filepath[1]).split('.')[0]
        if not data_name in datasets:
            datasets[data_name] = dict()
            Tranges[data_name] = (0, len(vector))
        datasets[data_name][algo_name] = events
    return(datasets, Tranges)

def f1_func(p, r):
    """
    Compute the f1 function
    :param p: precision numeric value
    :param r: recall numeric value
    :return: f1 numeric value
    """
    return(2*p*r/(p+r))


#----------------- metrics ----------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sklearn


def test_events(events):
    """
    Verify the validity of the input events
    :param events: list of events, each represented by a couple (start, stop)
    :return: None. Raise an error for incorrect formed or non ordered events
    """
    if type(events) is not list:
        raise TypeError('Input `events` should be a list of couples')
    if not all([type(x) is tuple for x in events]):
        raise TypeError('Input `events` should be a list of tuples')
    if not all([len(x) == 2 for x in events]):
        raise ValueError(
            'Input `events` should be a list of couples (start, stop)')
    if not all([x[0] <= x[1] for x in events]):
        raise ValueError(
            'Input `events` should be a list of couples (start, stop) with start <= stop')
    if not all([events[i][1] < events[i+1][0] for i in range(len(events) - 1)]):
        raise ValueError(
            'Couples of input `events` should be disjoint and ordered')


def pr_from_events(events_pred, events_gt, Trange):
    """
    Compute the affiliation metrics including the precision/recall in [0,1],
    along with the individual precision/recall distances and probabilities

    :param events_pred: list of predicted events, each represented by a couple
    indicating the start and the stop of the event
    :param events_gt: list of ground truth events, each represented by a couple
    indicating the start and the stop of the event
    :param Trange: range of the series where events_pred and events_gt are included,
    represented as a couple (start, stop)
    :return: dictionary with precision, recall, and the individual metrics
    """
    # testing the inputs
    test_events(events_pred)
    test_events(events_gt)

    # other tests
    minimal_Trange = infer_Trange(events_pred, events_gt)
    if not Trange[0] <= minimal_Trange[0]:
        raise ValueError('`Trange` should include all the events')
    if not minimal_Trange[1] <= Trange[1]:
        raise ValueError('`Trange` should include all the events')

    if len(events_gt) == 0:
        raise ValueError('Input `events_gt` should have at least one event')

    if has_point_anomalies(events_pred) or has_point_anomalies(events_gt):
        raise ValueError('Cannot manage point anomalies currently')

    if Trange is None:
        # Set as default, but Trange should be indicated if probabilities are used
        raise ValueError(
            'Trange should be indicated (or inferred with the `infer_Trange` function')

    E_gt = get_all_E_gt_func(events_gt, Trange)
    aff_partition = affiliation_partition(events_pred, E_gt)

    # Computing precision distance
    d_precision = [affiliation_precision_distance(
        Is, J) for Is, J in zip(aff_partition, events_gt)]

    # Computing recall distance
    d_recall = [affiliation_recall_distance(
        Is, J) for Is, J in zip(aff_partition, events_gt)]

    # Computing precision
    p_precision = [affiliation_precision_proba(
        Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]

    # Computing recall
    p_recall = [affiliation_recall_proba(
        Is, J, E) for Is, J, E in zip(aff_partition, events_gt, E_gt)]

    if _len_wo_nan(p_precision) > 0:
        p_precision_average = _sum_wo_nan(
            p_precision) / _len_wo_nan(p_precision)
    else:
        p_precision_average = p_precision[0]  # math.nan
    p_recall_average = sum(p_recall) / len(p_recall)

    dict_out = dict({'precision': p_precision_average,
                     'recall': p_recall_average,
                     'individual_precision_probabilities': p_precision,
                     'individual_recall_probabilities': p_recall,
                     'individual_precision_distances': d_precision,
                     'individual_recall_distances': d_recall})
    return (dict_out)


def produce_all_results():
    """
    Produce the affiliation precision/recall for all files
    contained in the `data` repository
    :return: a dictionary indexed by data names, each containing a dictionary
    indexed by algorithm names, each containing the results of the affiliation
    metrics (precision, recall, individual probabilities and distances)
    """
    datasets, Tranges = read_all_as_events()  # read all the events in folder `data`
    results = dict()
    for data_name in datasets.keys():
        results_data = dict()
        for algo_name in datasets[data_name].keys():
            if algo_name != 'groundtruth':
                results_data[algo_name] = pr_from_events(datasets[data_name][algo_name],
                                                         datasets[data_name]['groundtruth'],
                                                         Tranges[data_name])
        results[data_name] = results_data
    return (results)


import pandas as pd
def affiliation_evaluate(results_storage, metrics, labels, score, **args):
    if "affiliation" in metrics:
        results = []
        for thre in args['affiliation']:
            result = {}
            pred = (score > thre).astype(int)
            accuracy = sklearn.metrics.accuracy_score(labels, pred)
            events_label = convert_vector_to_events(labels)
            events_pred = convert_vector_to_events(pred)
            Trange = (0, len(pred))
            affiliation_metrics = pr_from_events(events_pred, events_label, Trange)
            result['Affiliation_thre'] = thre
            result['Affiliation_ACC'] = accuracy
            result['Affiliation_P'] = P = affiliation_metrics['precision']
            result['Affiliation_R'] = R = affiliation_metrics['recall']
            result['Affiliation_F1'] = 2 * P * R / (P + R)
            results.append(pd.DataFrame([result]))
        results_storage['affiliation'] = pd.concat(results, axis=0).reset_index(drop=True)



# ---------------------------- aucvus------------  
import numpy as np
from sklearn import metrics


class metricor:
    def metric_AUC(self, label, score):
        return metrics.roc_auc_score(label, score)

    def metric_PR(self, label, score):
        return metrics.average_precision_score(label, score)

    def range_convers_new(self, label):
        '''
        input: arrays of binary values 
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        L = []
        i = 0
        j = 0 
        while j < len(label):
            # print(i)
            while label[i] == 0:
                i+=1
                if i >= len(label):  #?
                    break			 #?
            j = i+1
            # print('j'+str(j))
            if j >= len(label):
                if j==len(label):
                    L.append((i,j-1))

                break
            while label[j] != 0:
                j+=1
                if j >= len(label):
                    L.append((i,j-1))
                    break
            if j >= len(label):
                break
            L.append((i, j-1))
            i = j
        return L

    def extend_postive_range(self, x, window=5):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0] 
            e = L[k][1] 
            
            
            x1 = np.arange(e,min(e+window//2,length))
            label[x1] += np.sqrt(1 - (x1-e)/(window))
            
            x2 = np.arange(max(s-window//2,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(window))
            
        label = np.minimum(np.ones(length), label)
        return label

    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0] 
            e = L[k][1] 
            
            l0 = int((e-s+1)*percentage)
            
            x1 = np.arange(e,min(e+l0,length))
            label[x1] += np.sqrt(1 - (x1-e)/(2*l0))
            
            x2 = np.arange(max(s-l0,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(2*l0))
            
        label = np.minimum(np.ones(length), label)
        return label

    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        product = labels * pred
        
        TP = np.sum(product)
        
        # recall = min(TP/P,1)
        P_new = (P+np.sum(labels))/2      # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP/P_new,1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))
        
        
        existence = 0
        for seg in L:
            if np.sum(product[seg[0]:(seg[1]+1)])>0:
                existence += 1
                
        existence_ratio = existence/len(L)
        # print(existence_ratio)
        
        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall*existence_ratio
        
        FP = np.sum(pred) - TP
        # TN = np.sum((1-pred) * (1-labels))
        
        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP/N_new
        
        Precision_RangeAUC = TP/np.sum(pred)
        
        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC

    def RangeAUC(self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type='window'):
        # AUC_type='window'/'percentage'
        score_sorted = -np.sort(-score)
        
        P = np.sum(labels)
        # print(np.sum(labels))
        if AUC_type=='window':
            labels = self.extend_postive_range(labels, window=window)
        else:   
            labels = self.extend_postive_range_individual(labels, percentage=percentage)
        
        # print(np.sum(labels))
        L = self.range_convers_new(labels)
        TF_list = np.zeros((252,2))
        Precision_list = np.ones(251)
        j=0
        for i in np.linspace(0, len(score)-1, 250).astype(int):
            threshold = score_sorted[i]
            # print('thre='+str(threshold))
            pred = score>= threshold
            TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P,L)
            j+=1
            TF_list[j]=[TPR,FPR]
            Precision_list[j]=(Precision)
            
        TF_list[j+1]=[1,1]
                
        width = TF_list[1:,1] - TF_list[:-1,1]
        height = (TF_list[1:,0] + TF_list[:-1,0])/2
        AUC_range = np.dot(width,height)

        width_PR = TF_list[1:-1,0] - TF_list[:-2,0]
        height_PR = (Precision_list[1:] + Precision_list[:-1])/2
        AP_range = np.dot(width_PR,height_PR)
        
        if plot_ROC:
            return AUC_range, AP_range, TF_list[:,1], TF_list[:,0], Precision_list
        
        return AUC_range

    def new_sequence(self, label, sequence_original, window):
        a = max(sequence_original[0][0] - window // 2, 0)
        sequence_new = []
        for i in range(len(sequence_original) - 1):
            if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
                sequence_new.append((a, sequence_original[i][1] + window // 2))
                a = sequence_original[i + 1][0] - window // 2
        sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
        return sequence_new
    
    def sequencing(self, x, L, window=5):
        label = x.copy().astype(float)
        length = len(label)

        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        label = np.minimum(np.ones(length), label)
        return label
    
    # TPR_FPR_window
    def RangeAUC_volume_opt(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            N_pred[k] = np.sum(pred)

        for window in window_3d:

            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                threshold = score_sorted[i]
                pred = score >= threshold
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                    if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                TP = 0
                N_labels = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio
                N_new = len(labels) - P_new
                FPR = FP / N_new

                Precision = TP / N_pred[j]

                j += 1
                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = (AUC_range)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]

            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = AP_range

        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)

    def RangeAUC_volume_opt_mem(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        p = np.zeros((thre, len(score)))

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score >= threshold
            p[k] = pred
            N_pred[k] = np.sum(pred)

        for window in window_3d:
            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * p[j][seg[0]:seg[1] + 1]
                    if (p[j][seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                N_labels = 0
                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], p[j][seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio

                N_new = len(labels) - P_new
                FPR = FP / N_new
                Precision = TP / N_pred[j]
                j += 1

                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]
            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            width = TF_list[1:, 1] - TF_list[:-1, 1]
            height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
            AUC_range = np.dot(width, height)
            auc_3d[window] = (AUC_range)

            width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
            height_PR = Precision_list[1:]
            AP_range = np.dot(width_PR, height_PR)
            ap_3d[window] = (AP_range)
        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)


import pandas as pd
def auc_vus_evaluate(results_storage, metrics, labels, score, version='opt', thre=250, **args):
    if "auc" in metrics:
        grader = metricor()
        result = {}
        AUC_ROC = grader.metric_AUC(labels, score)
        AUC_PR = grader.metric_PR(labels, score)
        result['AUC_ROC'] = AUC_ROC
        result['AUC_PR'] = AUC_PR
        results_storage['auc'] = pd.DataFrame([result])
    if "r_auc" in metrics:
        slidingWindow = args.get('slidingWindow', 100)  # default set 100
        grader = metricor()
        result = {}
        R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
        result['R_AUC_ROC'] = R_AUC_ROC
        result['R_AUC_PR'] = R_AUC_PR
        results_storage['r_auc'] = pd.DataFrame([result])
    if "vus" in metrics:
        slidingWindow = args.get('slidingWindow', 100)  # default set 100
        result = {}
        if version == 'opt_mem':
            tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume_opt_mem(
                labels_original=labels, score=score, windowSize=slidingWindow, thre=thre
            )
        else:
            tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume_opt(
                labels_original=labels, score=score, windowSize=slidingWindow, thre=thre
            )

        result['VUS_ROC'] = avg_auc_3d
        result['VUS_PR'] = avg_ap_3d
        results_storage['vus'] = pd.DataFrame([result])



#--------------------------------------------f1 --------------------------------

import numpy as np
import sklearn
import sklearn.preprocessing

def adjustment(gt, pred):
    adjusted_pred = np.array(pred)
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and adjusted_pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if adjusted_pred[j] == 0:
                        adjusted_pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if adjusted_pred[j] == 0:
                        adjusted_pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            adjusted_pred[i] = 1
    return adjusted_pred


class metricor_f1:
    def __init__(self, bias = 'flat'):
        self.bias = bias
        self.eps = 1e-15

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start +AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue +=  bi
            if i in p:
                MyValue += bi
        return MyValue/MaxValue
    
    def Cardinality_factor(self, Anomolyrange, Prange):
        score = 0
        start = Anomolyrange[0]
        end = Anomolyrange[1]
        for i in Prange:
            if i[0] >= start and i[0] <= end:
                score +=1
            elif start >= i[0] and start <= i[1]:
                score += 1
            elif end >= i[0] and end <= i[1]:
                score += 1
            elif start >= i[0] and end <= i[1]:
                score += 1
        if score == 0:
            return 0
        else:
            return 1/score
    
    def b(self, i, length):
        bias = self.bias
        if bias == 'flat':
            return 1
        elif bias == 'front-end bias':
            return length - i + 1
        elif bias == 'back-end bias':
            return i
        else:
            if i <= length/2:
                return i
            else:
                return length - i + 1
    
    def metric_RF1(self, label, preds):
        Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha=0.2)
        Rprecision = self.range_recall_new(preds, label, 0)[0]
        if Rprecision + Rrecall==0:
            RF1=0
        else:
            RF1 = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)
        return Rprecision, Rrecall, RF1
    
    def range_recall_new(self, labels, preds, alpha):
        p = np.where(preds == 1)[0]    # positions of predicted label==1
        range_pred = self.range_convers_new(preds)
        range_label = self.range_convers_new(labels)

        Nr = len(range_label)    # total # of real anomaly segments

        ExistenceReward = self.existence_reward(range_label, preds)


        OverlapReward = 0
        for i in range_label:
            OverlapReward += self.w(i, p) * self.Cardinality_factor(i, range_pred)


        score = alpha * ExistenceReward + (1-alpha) * OverlapReward
        if Nr != 0:
            return score/Nr, ExistenceReward/Nr, OverlapReward/Nr
        else:
            return 0,0,0

    def range_convers_new(self, label):
        '''
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        anomaly_ends, = np.where(np.diff(label) == -1)
        if len(anomaly_ends):
            if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
                # we started with an anomaly, so the start of the first anomaly is the start of the labels
                anomaly_starts = np.concatenate([[0], anomaly_starts])
        if len(anomaly_starts):
            if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
                # we ended on an anomaly, so the end of the last anomaly is the end of the labels
                anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
        return list(zip(anomaly_starts, anomaly_ends))
    
    def existence_reward(self, labels, preds):
        '''
        labels: list of ordered pair
        preds predicted data
        '''

        score = 0
        for i in labels:
            if preds[i[0]:i[1]+1].any():
                score += 1
        return score
    
    def _get_events(self, y_test, outlier=1, normal=0):
        events = dict()
        label_prev = normal
        event = 0  # corresponds to no event
        event_start = 0
        for tim, label in enumerate(y_test):
            if label == outlier:
                if label_prev == normal:
                    event += 1
                    event_start = tim
            else:
                if label_prev == outlier:
                    event_end = tim - 1
                    events[event] = (event_start, event_end)
            label_prev = label

        if label_prev == outlier:
            event_end = tim - 1
            events[event] = (event_start, event_end)
        return events

    def metric_EventF1PA(self, label, preds):
        from sklearn.metrics import precision_score
        true_events = self._get_events(label)

        tp = np.sum([preds[start:end + 1].any() for start, end in true_events.values()])
        fn = len(true_events) - tp
        rec_e = tp/(tp + fn)
        prec_t = precision_score(label, preds)
        EventF1PA1 = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

        return prec_t, rec_e, EventF1PA1


import pandas as pd
def f1_evaluate(results_storage, metrics, labels, score, **args):
    if "best_f1" in metrics:
        result = {}
        Ps, Rs, thres = sklearn.metrics.precision_recall_curve(labels, score)
        F1s = (2 * Ps * Rs) / (Ps + Rs)
        best_F1_index = np.argmax(F1s[np.isfinite(F1s)])
        best_thre = thres[best_F1_index]
        pred = (score > best_thre).astype(int)
        best_acc = sklearn.metrics.accuracy_score(labels, pred)
        result['thre_best'] = best_thre
        result['ACC_best'] = best_acc
        result['P_best'] = Ps[best_F1_index] 
        result['R_best'] = Rs[best_F1_index] 
        result['F1_best'] = F1s[best_F1_index]
        results_storage['best_f1'] = pd.DataFrame([result])
    if "f1_raw" in metrics:
        results = []
        for thre in args['f1_raw']:
            result = {}
            pred = (score > thre).astype(int)
            accuracy = sklearn.metrics.accuracy_score(labels, pred)
            P, R, F1, _ = sklearn.metrics.precision_recall_fscore_support(labels, pred, average="binary")
            result['thre_raw'] = thre
            result['ACC_raw'] = accuracy
            result['P_raw'] = P 
            result['R_raw'] = R 
            result['F1_raw'] = F1
            results.append(pd.DataFrame([result]))
        results_storage['f1_raw'] = pd.concat(results, axis=0).reset_index(drop=True)
    if "f1_pa" in metrics:
        results = []
        for thre in args['f1_pa']:
            result = {}
            pred = (score > thre).astype(int)
            adjusted_pred = adjustment(labels, pred)
            accuracy = sklearn.metrics.accuracy_score(labels, adjusted_pred)
            P, R, F1, _ = sklearn.metrics.precision_recall_fscore_support(labels, adjusted_pred, average="binary")
            result['thre_PA'] = thre
            result['ACC_PA'] = accuracy
            result['P_PA'] = P 
            result['R_PA'] = R 
            result['F1_PA'] = F1
            results.append(pd.DataFrame([result]))
        results_storage['f1_pa'] = pd.concat(results, axis=0).reset_index(drop=True)
    if "f1_r" in metrics:
        results = []
        for thre in args['f1_r']:
            result = {}
            pred = (score > thre).astype(int)
            accuracy = sklearn.metrics.accuracy_score(labels, pred)
            PR, RR, F1R= metricor_f1().metric_RF1(labels, pred)
            result['thre_r'] = thre
            result['ACC_r'] = accuracy
            result['P_r'] = PR
            result['R_r'] = RR 
            result['F1_r'] = F1R
            results.append(pd.DataFrame([result]))
        results_storage['f1_r'] = pd.concat(results, axis=0).reset_index(drop=True)
    if "f1_event" in metrics:
        results = []
        for thre in args['f1_event']:
            result = {}
            pred = (score > thre).astype(int)
            accuracy = sklearn.metrics.accuracy_score(labels, pred)
            PE, RE, F1E= metricor().metric_EventF1PA(labels, pred)
            result['thre_event'] = thre
            result['ACC_event'] = accuracy
            result['P_event'] = PE
            result['R_event'] = RE 
            result['F1_event'] = F1E
            results.append(pd.DataFrame([result]))
        results_storage['f1_event'] = pd.concat(results, axis=0).reset_index(drop=True)



#------------------------- pate  evaluator-----------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os


class Evaluator():
    def __init__(self, gt, anomaly_score, save_path):
        """
        input:
            gt: np.ndarray[int],
            anomaly_score: np.ndarray[float],
            save_path: str
        """
        self.gt = gt
        self.anomaly_score = anomaly_score
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def evaluate(self, metrics, merge=False, verbose=True, **metrics_args):
        """
        support metric: 'affiliation', 'auc', 'r_auc', 'vus', 'f1_raw', 'f1_pa', 'best_f1', 'f1_r', 'f1_event', 'pate', 'accomplish_UCR'

        input:
            metrics: List[str], e.g. ['affiliation', 'auc', 'r_auc', 'vus', 'f1_raw', 'f1_pa', 'best_f1', 'f1_r', 'f1_event', 'pate', 'accomplish_UCR']
            metrics_args: Dict[str, args], e.g. {'affiliation': [0.01, 0.02], 'f1_raw': [0.1, 0.2], ..., 'sliddingWindow': 100}
            merge: bool, if True: merge all results from different metrics
        output:
            results_storage: Dict[str, Dict[str, List[float]]]
        """
        results_storage = {}
        # metrics
        f1.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        affiliation.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        auc_vus.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        pate.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        accomplish_UCR.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        # save
        self._save_csv(results_storage, metrics, merge=merge, verbose=verbose, **metrics_args)
        return results_storage
    
    def _save_csv(self, results_storage, metrics, merge, verbose, **metrics_args):
        if merge:
            df = pd.concat([pd.DataFrame.from_dict(results_storage[metric]) for metric in metrics], axis=1)
            if verbose: print(df)
            df.to_csv(f'{self.save_path}/_results.csv', index=0)
        else:
            for metric in metrics:
                df = pd.DataFrame.from_dict(results_storage[metric])
                if verbose: print(df)
                df.to_csv(f'{self.save_path}/_{metric}.csv', index=0)

    def find_thres(self, method, verbose=True, **args):
        """
        support method: 'prior_anomaly_rate', 'spot'  

        input:  
            method: str   
            args:  
            method == 'prior_anomaly_rate': require pAR: List[float]  
            method == 'spot': require init_score: np.ndarray[float]; q: List[float]  
        output:  
            thresholds: List[float]  
        
        example:  
            thres = evaluator.find_thres(method='prior_anomaly_rate', pAR=[0.05, 0.1])  
            thres = evaluator.find_thres(method='spot', init_score=init_score, q=[0.1, 0.2])  
        """
        if method == 'prior_anomaly_rate':
            thresholds = [np.percentile(self.anomaly_score, 100 * (1-pAR)) for pAR in args['pAR']]
            self._save_thres_info(args['pAR'], thresholds, method, verbose)
        elif method == 'spot':
            from .spot import SPOT
            thresholds = []
            for q in args['q']:
                s = SPOT(q)
                s.fit(args['init_score'], self.anomaly_score)
                s.initialize(verbose=False)
                ret = s.run()
                thresholds.append(np.mean(ret['thresholds']))
            self._save_thres_info(args['q'], thresholds, method, verbose)
            
        return thresholds
    
    def _save_thres_info(self, arg1, arg2, method, verbose):
        thres_info = pd.DataFrame(np.stack([arg1, arg2], axis=1), columns=['hyper-parameter', 'threshold'])
        if verbose: print(thres_info)
        thres_info.to_csv(f'{self.save_path}/_{method}_thres_info.csv', index=0)
    
    def vis_anomaly_intervals_all(self, series=None, start=None, end=None):
        plt.rcParams.update({'font.size': 14})
        
        if start is None: start = 0
        if end is None: end = len(self.gt)

        if series is None: nc = 1
        else: 
            nc = series.shape[-1]
            assert len(series) == len(self.anomaly_score)
            series = series[start:end]
            
        gt = self.gt[start:end]
        anomaly_score = self.anomaly_score[start:end]
        as_min, as_max = anomaly_score.min(0), anomaly_score.max(0)
        anomaly_score = (anomaly_score - as_min) / (as_max - as_min)

        borders = self._find_borders(gt)
        n_anomalies = len(borders)

        vis_path = os.path.join(self.save_path, "vis")
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
    
        for c in range(nc):
            fig, ax1 = plt.subplots(figsize=(20, 5))
            # series
            if series is not None:
                ax1.plot(range(start, end), series[:, c], linewidth=1, label='series', color='#5861AC')
                ax1.set_ylabel('Series value')
            # anomaly score
            ax2 = ax1.twinx()
            # ax2.plot(range(start, end), anomaly_score, linewidth=0.2, color='#72C3A3')
            ax2.fill_between(range(start, end), anomaly_score, 0, label='anomlay score', color='#72C3A3', alpha=0.3)
            ax2.set_ylabel('Anomaly score')
            ax2.set_xlabel('Time step')
            # abnormly interval
            for i in range(n_anomalies):
                if i == 0: 
                    plt.axvspan(xmin=borders[i][0]+start, xmax=borders[i][1]+start, ymin=0, ymax=1, label='anomaly interval', facecolor='r', alpha=0.3)
                else:
                    plt.axvspan(xmin=borders[i][0]+start, xmax=borders[i][1]+start, ymin=0, ymax=1, facecolor='r', alpha=0.2)
            
            fig.legend(loc='upper left', bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
            plt.savefig(f'{vis_path}/vis_{start}-{end}_c{c}.pdf', bbox_inches='tight')
            plt.clf()

    def vis_anomaly_intervals_each(self, series=None, max_span=100, max_anomalies=1):
        plt.rcParams.update({'font.size': 14})

        if series is None: nc = 1
        else: 
            nc = series.shape[-1]
            assert len(series) == len(self.anomaly_score)
            
        borders = self._find_borders(self.gt)
        n_anomalies = len(borders)

        if max_anomalies is None: anomalies_list = range(n_anomalies)
        elif isinstance(max_anomalies, int): anomalies_list = range(min(n_anomalies, max_anomalies))

        vis_path = os.path.join(self.save_path, "vis_anorm")
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        for c in range(nc):
            for i in anomalies_list:
                start = 0
                end = len(self.anomaly_score)
                if i:
                    start = borders[i-1][1]
                if i+1 < n_anomalies:
                    end = borders[i+1][0]-1
                
                if max_span is not None:
                    start = max(start, borders[i][0]-max_span)
                    end = min(end, borders[i][1]+max_span)

                fig, ax1 = plt.subplots(figsize=(6, 5))
                # series
                if series is not None:
                    series_i = series[start:end, c]
                    ax1.plot(range(start, end), series_i, linewidth=1, label='series', color='#5861AC')
                    ax1.set_ylabel('Series value')
                # anomaly score
                ax2 = ax1.twinx()
                score_i = self.anomaly_score[start:end]
                as_min, as_max = score_i.min(0), score_i.max(0)
                score_i = (score_i - as_min) / (as_max - as_min)
                # ax2.plot(range(start, end), score_i, linewidth=0.2, color='#72C3A3')
                ax2.fill_between(range(start, end), score_i, 0, label='anomlay score', color='#72C3A3', alpha=0.3)
                ax2.set_ylabel('Anomaly score')
                ax2.set_xlabel('Time step')

                # abnormly interval
                plt.axvspan(xmin=borders[i][0], xmax=borders[i][1], ymin=0, ymax=1, label='anomaly interval', facecolor='r', alpha=0.2)

                fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
                plt.savefig(f'{vis_path}/anorm{i}_{borders[i][0]}-{borders[i][1]}_c{c}.pdf', bbox_inches='tight')
                plt.clf()
                
    def _find_borders(self, gt):
        """
        return anomaly intervals: [[start1, end1)...[startn, endn)]
        """
        borders = []
        s = 0
        while True:
            if gt[s]==1:
                e = s
                while gt[e]==1:
                    if e == len(gt): break
                    e += 1
                borders.append([s, e])
                s = e+1
            else:
                s += 1
            if s >= len(gt): break
        return borders


#-------------------- pate spot ---------------------------------------

from math import log, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'


class SPOT:
    def __init__(self, q=1e-4):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, verbose=True):
        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]

        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        zeros = np.concatenate((left_zeros, right_zeros))

        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True):
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        th = []
        alarm = []
        for i in tqdm(range(self.data.size)):

            if self.data[i] > self.extreme_quantile:
                if with_alarm:
                    alarm.append(i)
                else:
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)

            elif self.data[i] > self.init_threshold:
                self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                self.Nt += 1
                self.n += 1

                g, s, l = self._grimshaw()
                self.extreme_quantile = self._quantile(g, s)
            else:
                self.n += 1

            th.append(self.extreme_quantile)

        return {'thresholds': th, 'alarms': alarm}

    def plot(self, run_results, with_alarm=True):
        x = range(self.data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if 'thresholds' in K:
            th = run_results['thresholds']
            th_fig, = plt.plot(x, th, color=deep_saffron, lw=2, ls='dashed')
            fig.append(th_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            al_fig = plt.scatter(alarm, self.data[alarm], color='red')
            fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig


#------------------------------------------------------------------------------------------


def data_provider(root_path, datasets, batch_size, win_size=100, step=100, flag="train", percentage=1):
    if flag == "train": shuffle = True
    else: shuffle = False
    print(f"loading {datasets}({flag}) percentage: {percentage*100}% ...", end="")
    file_paths, train_lens = read_meta(root_path=root_path, dataset=datasets)
    discrete_channels = None
    if datasets == "MSL": discrete_channels = range(1, 55)
    if datasets == "SMAP": discrete_channels = range(1, 25)
    if datasets == "SWAT": discrete_channels =  [2,4,9,10,11,13,15,19,20,21,22,29,30,31,32,33,42,43,48,50]
    data_set = TrainSegLoader(file_paths, train_lens, win_size, step, flag, percentage, discrete_channels)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    print("done!")
    return data_set, data_loader

class TrainSegLoader(Dataset):
    def __init__(self, data_path, train_length, win_size, step, flag="train", percentage=0.1, discrete_channels=None):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        # 1.read data
        data = read_data(data_path)
        # 2.train
        train_data = data.iloc[:train_length, :]
        train_data, train_label =  (
            train_data.loc[:, train_data.columns != "label"].to_numpy(),
            train_data.loc[:, ["label"]].to_numpy(),
        )
        # 3.test
        test_data = data.iloc[train_length:, :]
        test_data, test_label =  (
            test_data.loc[:, test_data.columns != "label"].to_numpy(),
            test_data.loc[:, ["label"]].to_numpy(),
        )
        # 4.process
        if discrete_channels is not None:
            train_data = np.delete(train_data, discrete_channels, axis=-1)
            test_data = np.delete(test_data, discrete_channels, axis=-1)
        
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        if flag == "init":
            self.init = train_data
            self.init_label = train_label
        else:
            train_end = int(len(train_data) * 0.8)
            train_start = int(train_end*(1-percentage))
            self.train = train_data[train_start:train_end]
            self.train_label = train_label[train_start:train_end]
            self.val = train_data[train_end:]
            self.val_label = train_label[train_end:]
            self.test = test_data
            self.test_label = test_label

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "init":
            return (self.init.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index, eps=1):
        index = index * self.step
        if self.flag == "train":           
            return np.float32(self.train[index: index + self.win_size]), np.float32(self.train_label[index: index + self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index: index + self.win_size]), np.float32(self.val_label[index: index + self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index: index + self.win_size]), np.float32(self.test_label[index: index + self.win_size])
        elif self.flag == "init":
            return np.float32(self.init[index: index + self.win_size]), np.float32(self.init_label[index: index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size: index// self.step * self.win_size+ self.win_size]), np.float32(self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])

def read_data(path: str, nrows=None) -> pd.DataFrame:
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values
    all_points = data.shape[0]
    columns = data.columns
    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()
    is_univariate = n_points == all_points
    n_cols = all_points // n_points
    df = pd.DataFrame()
    cols_name = data["cols"].unique()
    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    else:
        df[cols_name[0]] = data.iloc[:, 0]
    if label_exists:
        last_col_name = df.columns[-1]
        df.rename(columns={last_col_name: "label"}, inplace=True)
    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]
    return df

def read_meta(root_path, dataset):
    meta_path = root_path + "/DETECT_META.csv"
    meta = pd.read_csv(meta_path)
    meta = meta.query(f'file_name.str.contains("{dataset}")', engine="python")
    file_paths = root_path + f"/data/{meta.file_name.values[0]}"
    train_lens = meta.train_lens.values[0]
    return file_paths, train_lens


#--------------------------------------------------exp--------------------------
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import math
import json


warnings.filterwarnings('ignore')

class Exp_Anomaly_Detection():
    def __init__(self, args, id=None):
        self.args = args
        self.device = self._acquire_device()

        if self.args.data_origin == "UCR":
            self.id = id
            self.model_configs = Configs(model_configs_0)
            self.train_configs = Configs(train_configs)
            self.model_save_path = os.path.join(self.args.configs_path, self.args.data_origin, f"{self.args.data}/checkpoints_{id}")    # save model checkpoints
            self.rst_save_path = os.path.join(self.args.save_path, self.args.data_origin, f"{self.args.data}/results_{id}")                                               # save results
        else:
            self.model_configs = Configs(model_configs_0)
            self.train_configs = Configs(train_configs)
            self.model_save_path = os.path.join(self.args.configs_path, self.args.data, f"checkpoints_{id}")          # save model checkpoints
            self.rst_save_path = os.path.join(self.args.save_path, self.args.data, f"results_{id}")                   # save results

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.exists(self.rst_save_path):
            os.makedirs(self.rst_save_path)

        self.model = self._build_model().to(self.device)
           
    def _build_model(self):
        model = Basic_CrossAD(self.model_configs)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _basic(self):
        try:
            import pandas as pd
            from thop import profile
            from thop import clever_format
            dummy_input = torch.rand(1, self.model_configs.seq_len, 1).to(self.device)
            flops, params = profile(self.model, inputs=(dummy_input,None,None,None))
            flops, params = clever_format([flops, params], '%.3f')
            basic_info = pd.DataFrame([{'params': params, 'flops': flops}])
            basic_info.to_csv(self.model_save_path + '/basic_info.csv', index=False)
            print(basic_info)
        except ImportError:
            print("Warning: 'thop' module not found. Skipping model profiling.")

    def _acquire_device(self):
        device = torch.device('cpu')
        print('Use CPU')
        return device

    def _get_data(self, flag, step=None):
        win_size = self.model_configs.seq_len
        if step is None:
            step = win_size
        batch_size = self.train_configs.batch_size
        data_set, data_loader = data_provider(
            root_path=self.args.root_path, 
            datasets=self.args.data, 
            batch_size=batch_size, 
            win_size=win_size,
            step=step, 
            flag=flag, 
        )
        return data_set, data_loader       

    def _select_optimizer(self):
        if self.train_configs.optim == "adam":
            model_optim = optim.Adam(self.model.parameters(), lr=self.train_configs.learning_rate)
        elif self.train_configs.optim == "adamw":
            model_optim = optim.AdamW(self.model.parameters(), lr=self.train_configs.learning_rate)
        return model_optim

    def _adjust_learning_rate(self, optimizer, epoch, train_configs, verbose=True, **other_args):
        if train_configs.lradj == 'type1':
            lr_adjust = {epoch: train_configs.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif train_configs.lradj == 'type2':
            lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
        elif train_configs.lradj == 'type3':
            lr_adjust = {epoch: train_configs.learning_rate if epoch < 3 else train_configs.learning_rate * (0.9 ** ((epoch - 3) // 1))}
        elif train_configs.lradj == "cosine":
            lr_adjust = {epoch: train_configs.learning_rate /2 * (1 + math.cos(epoch / train_configs.train_epochs * math.pi))}
        elif train_configs.lradj == '1cycle':
            scheduler = other_args['scheduler']
            lr_adjust = {epoch: scheduler.get_last_lr()[0]}
        
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if verbose: print('Updating learning rate to {}'.format(lr))
    
    def vali(self, vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                ms_loss, q_latent_distance = self.model(batch_x, None, None, None)
                loss = ms_loss
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        self._basic()

        _, train_loader = self._get_data(flag='train', step=TRAIN_STEP)
        _, vali_loader = self._get_data(flag='val')
        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.train_configs.patience, verbose=True)
        if self.train_configs.lradj == "1cycle":
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=model_optim, 
                                                    steps_per_epoch=train_steps, 
                                                    pct_start=self.train_configs.pct_start, 
                                                    epochs=self.train_configs.train_epochs, 
                                                    max_lr=self.train_configs.learning_rate
                                                    )

        time_now = time.time()
        for epoch in range(self.train_configs.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                ms_loss, q_latent_distance = self.model(batch_x, None, None, None)
                loss = ms_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_configs.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.train_configs.lradj == '1cycle':
                    self._adjust_learning_rate(model_optim, epoch + 1, self.train_configs, verbose=False, scheduler=scheduler)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, self.model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if self.train_configs.lradj != "1cycle":
                self._adjust_learning_rate(model_optim, epoch + 1, self.train_configs)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = self.model_save_path + '/checkpoint.pth'
        state_dict = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def test(self, **args):
        _, test_loader = self._get_data(flag='test')

        print('loading model...', end='')
        state_dict = torch.load(os.path.join(self.model_save_path, 'checkpoint.pth'), map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        print('done')

        self.model.eval()
        with torch.no_grad():
            # test set
            attens_energy = []
            test_labels = []
            # test_series = []
            for i, (batch_x, batch_y) in enumerate(test_loader):
                # test_series.append(batch_x)
                test_labels.append(batch_y)

                batch_x = batch_x.float().to(self.device)
                score, q_latent_distance = self.model.infer(batch_x, None, None, None)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

            attens_energy = np.concatenate(attens_energy, axis=0)                       # nb x t x c
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])          # nb*t x c
            test_energy = np.array(attens_energy)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)               # nb*t
            test_labels = np.array(test_labels)
            test_gt = test_labels.astype(int)
            # test_series = np.concatenate(test_series, axis=0)                         # nb x t x c
            # test_series = test_series.reshape(-1, test_series.shape[-1])              # nb*t x c
            # test_series = np.array(test_series)
        
        np.save(self.rst_save_path + '/a_gt.npy', test_gt)                              # nb*t
        np.save(self.rst_save_path + '/a_test_energy.npy', test_energy)                 # nb*t x c
        
        test_energy = np.mean(test_energy, axis=-1)                                     # nb*t

        evaluator = Evaluator(test_gt, test_energy, self.rst_save_path)
        evaluator.evaluate(metrics=self.args.metrics)                                   # not need threshold
        
    def evaluate_spot(self, **args):
        gt = np.load(self.rst_save_path + '/a_gt.npy')
        test_energy = np.load(self.rst_save_path + '/a_test_energy.npy')

        test_energy = np.mean(test_energy, axis=-1)                                     # nb*t

        evaluator = Evaluator(gt, test_energy, self.rst_save_path)

        # find threshold by spot
        _, init_loader = self._get_data(flag='init')

        print('loading model...', end='')
        state_dict = torch.load(os.path.join(self.model_save_path, 'checkpoint.pth'),
                        map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        print('done')

        self.model.eval()
        with torch.no_grad():
            attens_energy = []
            for i, (batch_x, batch_y) in enumerate(init_loader):
                batch_x = batch_x.float().to(self.device)
                score, q_latent_distance = self.model.infer(batch_x, None, None, None)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

            attens_energy = np.concatenate(attens_energy, axis=0)                       # nb x t x c
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])          # nb*t x c
            init_energy = np.array(attens_energy)
           
        init_energy = np.mean(init_energy, axis=-1)                                 # nb*t
            
        thresholds = evaluator.find_thres(method="spot", init_score=init_energy, q=args['t'])

        evaluator.evaluate(metrics=['affiliation'], affiliation=thresholds)
    
    def evaluate_UCR_accomplish(self, **args):
        _, test_loader = self._get_data(flag='test')

        print('loading model...', end='')
        self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, 'checkpoint.pth')), strict=False)
        print('done')

        self.model.eval()
        with torch.no_grad():
            # test set
            attens_energy = []
            test_labels = []
            test_series = []
            for i, (batch_x, batch_y) in enumerate(test_loader):
                test_series.append(batch_x)
                test_labels.append(batch_y)

                batch_x = batch_x.float().to(self.device)
                score, q_latent_distance = self.model.infer(batch_x, None, None, None)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

            attens_energy = np.concatenate(attens_energy, axis=0)                       # nb x t x c
            attens_energy = attens_energy.reshape(-1, attens_energy.shape[-1])          # nb*t x c
            test_energy = np.array(attens_energy)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)               # nb*t
            test_labels = np.array(test_labels)
            test_gt = test_labels.astype(int)
            test_series = np.concatenate(test_series, axis=0)                           # nb x t x c
            test_series = test_series.reshape(-1, test_series.shape[-1])                # nb*t x c
            test_series = np.array(test_series)
        
        np.save(self.rst_save_path + '/a_gt.npy', test_gt)                              # nb*t
        np.save(self.rst_save_path + '/a_test_energy.npy', test_energy)                 # nb*t x c

        test_energy = np.mean(test_energy, axis=-1)                                     # nb*t

        evaluator = Evaluator(test_gt, test_energy, self.rst_save_path)
        results_storage = evaluator.evaluate(metrics=['accomplish_UCR'])
        evaluator.vis_anomaly_intervals_all(test_series)

        filename = os.path.join(self.args.save_path, self.args.data_origin, 'ucr.csv')
        results = [self.args.data, results_storage['accomplish_UCR']['topk'].values[0], 
                   results_storage['accomplish_UCR']['total_len'].values[0], 
                   results_storage['accomplish_UCR']['aplha_quantile'].values[0], 
                   results_storage['accomplish_UCR']['3_alpha'].values[0],
                   results_storage['accomplish_UCR']['10_alpha'].values[0]]
        with open(filename, 'a') as f:
            f.write(','.join([str(result) for result in results]) + '\n')

    def analysis(self):
        self._basic()

   
class Configs:
    def __init__(self, json_path):
        with open(json_path) as f:
            configs = json.load(f)
            self.__dict__.update(configs)


class Configs:
    def __init__(self, cfg_input):
        if isinstance(cfg_input, str):
            with open(cfg_input) as f:
                configs = json.load(f)
            self.__dict__.update(configs)
        elif isinstance(cfg_input, dict):
            self.__dict__.update(cfg_input)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


#---------------- basic crossad ----------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import json






class Basic_CrossAD(nn.Module):
    def __init__(self, configs):
        super(Basic_CrossAD, self).__init__()
        seq_len=configs.seq_len
        patch_len=configs.patch_len
        d_model=configs.d_model
        ms_kernerls=configs.ms_kernels
        ms_method=configs.ms_method

        self.n_scales = len(ms_kernerls)
        self.ms_utils = MS_Utils(ms_kernerls, ms_method)
        self.pos_embedding = PositionalEmbedding(d_model)
        self.patch_embedding = PatchEmbedding(d_model, patch_len=patch_len, stride=patch_len, padding=(patch_len-1), dropout=0.)
        self.ms_t_lens = self.ms_utils._dummy_forward(seq_len)
        self.ms_p_lens = self.patch_embedding._dummy_forward(self.ms_t_lens)
        self.ms_t_lens_ = [PN * patch_len for PN in self.ms_p_lens]

        self.scale_ind_mask = self.ms_utils.scale_ind_mask(self.ms_p_lens) 
        if torch.cuda.is_available():
            self.scale_ind_mask = self.scale_ind_mask.cuda()


        self.next_scale_mask = self.ms_utils.next_scale_mask(self.ms_p_lens)
        if torch.cuda.is_available():
            self.next_scale_mask = self.next_scale_mask.cuda()


        if "batch" in configs.norm.lower():
            encoder_norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            decoder_norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            encoder_norm = nn.LayerNorm(d_model)
            decoder_norm = nn.LayerNorm(d_model)

        self.encoder=Encoder(
            layers=[
                EncoderLayer(
                    attention=AttentionLayer(ScaledDotProductAttention(attn_dropout=configs.attn_dropout), d_model=configs.d_model, n_heads=configs.n_heads, proj_dropout=configs.proj_dropout),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    norm=configs.norm,
                    dropout=configs.ff_dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=encoder_norm
        )
        self.decoder=Decoder(
            layers=[
                DecoderLayer(
                    self_attention=AttentionLayer(ScaledDotProductAttention(attn_dropout=configs.attn_dropout), d_model=configs.d_model, n_heads=configs.n_heads, proj_dropout=configs.proj_dropout),
                    cross_attention=AttentionLayer(ScaledDotProductAttention(attn_dropout=configs.attn_dropout), d_model=configs.d_model, n_heads=configs.n_heads, proj_dropout=configs.proj_dropout),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    norm=configs.norm,
                    dropout=configs.ff_dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=decoder_norm,
            projection=nn.Sequential(nn.Linear(configs.d_model, configs.patch_len), nn.Flatten(-2))
        )
        self.context_net=ContextNet(
            router = Router(seq_len=self.ms_t_lens[-1], n_vars=1, n_query=configs.n_query, topk=configs.topk),
            querys = nn.Parameter(torch.randn(configs.n_query, configs.query_len, configs.d_model)),
            extractor = Extractor(
                layers=[
                    ExtractorLayer(
                        cross_attention=AttentionLayer(ScaledDotProductAttention(attn_dropout=configs.attn_dropout), d_model=configs.d_model, n_heads=configs.n_heads, proj_dropout=configs.proj_dropout),
                        d_model=configs.d_model,
                        d_ff=configs.d_ff,
                        norm=configs.norm,
                        dropout=configs.ff_dropout,
                        activation=configs.activation
                    ) for _ in range(configs.m_layers)
                ],
                context_size=configs.bank_size,
                query_len=configs.query_len,
                d_model=configs.d_model,
                decay=configs.decay,
                epsilon=configs.epsilon
            )
        )

    def _forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        bs, t, c = x_enc.shape

        # CI
        x_enc = x_enc.permute(0, 2, 1)                                                          
        x_enc = x_enc.reshape(bs*c, t, 1)                                                       # x_enc: [bs*c x t x 1]
        router_input = x_enc

        # generate multi-scale x_enc
        ms_x_enc_list = self.ms_utils(x_enc)
        ms_gt = self.ms_utils.concat_sampling_list(ms_x_enc_list[1:] + [x_enc])                 # ms_gt: [bs*c x ms_t x 1]
        ms_gt = ms_gt.reshape(bs, c, -1)                                                        # ms_gt: [bs x c x ms_t]
        ms_gt = ms_gt.permute(0, 2, 1)                                                          # ms_gt: [bs x ms_t x c]

        # patch_embedding + pos_embedding
        x_enc = x_enc.permute(0, 2, 1)                                                          # x_enc: [bs*c x 1 x t]
        _, x_enc = self.patch_embedding(x_enc)                                                  # x_enc: [bs*c x pn x pl]
        for i in range(self.n_scales):
            x_enc_i = ms_x_enc_list[i]
            x_enc_i = x_enc_i.permute(0, 2, 1)                                                  # x_enc_i: [bs*c x 1 x t_i]
            x_enc_emb_i, _ = self.patch_embedding(x_enc_i)                                      # x_enc_emb_i: [bs*c x pn_i x d_model]
            ms_x_enc_list[i] = x_enc_emb_i
        ms_x_enc = self.ms_utils.concat_sampling_list(ms_x_enc_list)                            # ms_x_enc: [bs*c x ms_pn x d_model]

        pos_emb = self.pos_embedding(self.ms_p_lens[-1])                                        # pos_emb: [1 x pn x d_model]
        ms_pos_emb_list = self.ms_utils(pos_emb)
        ms_pos_emb = self.ms_utils.concat_sampling_list(ms_pos_emb_list)                        # ms_pos_emb: [1 x ms_pn x d_model]

        ms_x_enc = ms_x_enc + ms_pos_emb                                                        # ms_x_enc: [bs*c x ms_pn x d_model]

        # scale-independence encoder
        ms_x_enc, attn_weights = self.encoder(ms_x_enc, self.scale_ind_mask)                    # ms_x_enc_repr: [bs*c x ms_pn x d_model]

        # period context
        if self.training:
            query_latent_distances, context = self.context_net(router_input, ms_x_enc)              # context: [N*query_len x d_model]
            context = context.unsqueeze(0).expand(bs*c, -1, -1)                                     # context: [bs*c x N*query_len x d_model]
            query_latent_distances = query_latent_distances.reshape(bs, c, 1)                       # query_latent_distances: [bs x c x 1]
            query_latent_distances = query_latent_distances.permute(0, 2, 1)                        # query_latent_distances: [bs x 1 x c]
        else:
            query_latent_distances = torch.zeros(bs, 1, c)
            context = self.context_net.extractor.concat_context(self.context_net.extractor.context)
            context = context.unsqueeze(0).expand(bs*c, -1, -1)

        # next-scale decoder
        ms_x_enc_list = self.ms_utils.split_2_list(ms_x_enc, ms_t_lens=self.ms_p_lens, mode="encoder")
        ms_x_dec_list = self.ms_utils.up(ms_x_enc_list, ms_t_lens=self.ms_p_lens)
        ms_x_dec = self.ms_utils.concat_sampling_list(ms_x_dec_list)                                                # ms_x_dec_repr: [bs*c x up(ms_pn) x d_model]

        ms_x_dec, self_attn_weights, cross_attn_weights = self.decoder(ms_x_dec, context, self.next_scale_mask)     # ms_x_dec: [bs*c x up(ms_pn)*pl]
        ms_x_dec = ms_x_dec.reshape(bs*c, -1, 1)                                                                    # ms_x_dec: [bs*c x up(ms_pn)*pl x 1]

        ms_x_dec = ms_x_dec.reshape(bs, c, -1)                                                                      # ms_x_dec: [bs x c x up(ms_pn)*pl]
        ms_x_dec = ms_x_dec.permute(0, 2, 1)                                                                        # ms_x_dec: [bs x up(ms_pn)*pl x c]
        ms_x_dec_list = self.ms_utils.split_2_list(ms_x_dec, ms_t_lens=self.ms_t_lens_, mode="decoder")
        for i in range(len(ms_x_dec_list)):
            ms_x_dec_list[i] = ms_x_dec_list[i][:, :self.ms_t_lens[i+1]]
        ms_x_dec = self.ms_utils.concat_sampling_list(ms_x_dec_list)                                                # ms_x_dec: [bs x ms_t x c]
        
        return ms_gt, ms_x_dec, query_latent_distances
    
    def _ms_anomaly_score(self, ms_x_dec, ms_gt):
        ms_score = F.mse_loss(ms_x_dec, ms_gt, reduction="none")                                                    # score: [bs x ms_t x c]
        ms_score_list = self.ms_utils.split_2_list(ms_score, ms_t_lens=self.ms_t_lens, mode="decoder")              # score_list: [[bs x t1 x c] ... [bs x ti x c]]

        for i in range(len(ms_score_list)-1):
            loss_i = ms_score_list[i].permute(0, 2, 1)                                                              # [b x c x t_i]
            up_loss_i = F.interpolate(loss_i, size=ms_score_list[-1].shape[1], mode='linear').permute(0, 2, 1)      # [b x t x c]
            ms_score_list[-1] = ms_score_list[-1] + up_loss_i
            
        ms_score = ms_score_list[-1]                                                                                # [b x t x c]
        return ms_score

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        ms_gt, ms_x_dec, query_latent_distances = self._forward(x_enc, x_mark_enc, x_dec, x_mark_dec)               # [bs x ms_t x c]
        return F.mse_loss(ms_x_dec, ms_gt), torch.mean(query_latent_distances)

    def infer(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        ms_gt, ms_x_dec, query_latent_distances = self._forward(x_enc, x_mark_enc, x_dec, x_mark_dec)               # [bs x ms_t x c]                              
        return self._ms_anomaly_score(ms_x_dec, ms_gt), query_latent_distances                                      # [bs, t, c], [bs, 1, c]


class MS_Utils(nn.Module):
    def __init__(self, kernels, method="interval_sampling"):
        super().__init__()
        self.kernels = kernels
        self.method = method

    def concat_sampling_list(self, x_enc_sampling_list):
        return torch.concat(x_enc_sampling_list, dim=1)                                                             # [b x ms_t x -1]
    
    def split_2_list(self, ms_x_enc, ms_t_lens, mode="encoder"):
        if mode == "encoder":
            return list(torch.split(ms_x_enc, split_size_or_sections=ms_t_lens[:-1], dim=1))
        elif mode == "decoder":
            return list(torch.split(ms_x_enc, split_size_or_sections=ms_t_lens[1:], dim=1))
    
    def scale_ind_mask(self, ms_t_lens):
        L = sum(t_len for t_len in ms_t_lens[:-1])
        d = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(ms_t_lens[:-1])]).view(1, L, 1)
        dT = d.transpose(1, 2)
        return torch.where(d == dT, 0., -torch.inf).reshape(1, 1, L, L).contiguous().bool()
    
    def next_scale_mask(self, ms_t_lens):
        L = sum(t_len for t_len in ms_t_lens[1:])
        d = torch.cat([torch.full((t_len,), i) for i, t_len in enumerate(ms_t_lens[1:])]).view(1, L, 1)
        dT = d.transpose(1, 2)
        return torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L, L).contiguous().bool()
    
    def down(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)                                                                              # [b x c x t]
        x_enc_sampling_list = []
        for kernel in self.kernels:
            pad_x_enc = F.pad(x_enc, pad=(0,kernel-1), mode="replicate")
            x_enc_i = pad_x_enc.unfold(dimension=-1, size=kernel, step=kernel)                                      # [b x c x t_i x kernel]
            if self.method == "average_pooling":
                x_enc_i = torch.mean(x_enc_i, dim=-1)
            elif self.method == "interval_sampling":
                x_enc_i = x_enc_i[:, :, :, 0]
            x_enc_sampling_list.append(x_enc_i.permute(0, 2, 1))                                                    # [b x t_i x c]

        return x_enc_sampling_list

    def up(self, x_enc_sampling_list, ms_t_lens):
        for i in range(len(ms_t_lens)-1):
            x_enc = x_enc_sampling_list[i].permute(0, 2, 1)                                                         # [b x c x t]
            up_x_enc = F.interpolate(x_enc, size=ms_t_lens[i+1], mode='nearest').permute(0, 2, 1)                   # [b x t x c]
            x_enc_sampling_list[i] = up_x_enc
        return x_enc_sampling_list
    
    @torch.no_grad()
    def _dummy_forward(self, input_len):
        dummy_x = torch.ones((1, input_len, 1))
        dummy_sampling_list = self.down(dummy_x)
        ms_t_lens = []
        for i in range(len(dummy_sampling_list)):
            ms_t_lens.append(dummy_sampling_list[i].shape[1])
        ms_t_lens.append(input_len)
        return ms_t_lens
    
    def forward(self, x_enc):
        return self.down(x_enc)
    

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, t_len):
        return self.pe[:, :t_len]
    

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def _dummy_forward(self, input_lens):
        ms_p_lens = []
        for i, input_len in enumerate(input_lens):
            dummy_x = torch.ones((1, 1, input_len))
            dummy_x = self.padding_patch_layer(dummy_x)
            dummy_x = dummy_x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            ms_p_lens.append(dummy_x.shape[2])
        return ms_p_lens

    def forward(self, x):
        # do patching
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_patch = x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), x_patch

#----------------------------- context_block -------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class ContextNet(nn.Module):
    def __init__(self, router, querys, extractor):
        super().__init__()
        self.router = router
        self.querys = querys
        self.extractor = extractor
    
    def forward(self, x_enc, local_repr, mask=None):
        q_indices = self.router(x_enc)
        q = torch.einsum('bn,nqd->bqd', q_indices, self.querys)                                         # q: [bs x query_len x d_model]
        
        query_latent_distances, context = self.extractor(q, local_repr, mask)
        return query_latent_distances, context


class Router(nn.Module):
    def __init__(self, seq_len, n_vars, n_query, topk=5):
        super().__init__()
        self.k = topk
        self.fc = nn.Sequential(nn.Flatten(-2), nn.Linear(seq_len*n_vars, n_query))

    def forward(self, x):
        bs, t, c = x.shape
        # fft
        x_freq = torch.fft.rfft(x, dim=1, n=t)
        # topk
        _, indices = torch.topk(x_freq.abs(), self.k, dim=1)    # indices: [bs x k x c]                             
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing="ij")
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        mask = torch.zeros_like(x_freq, dtype=torch.bool)       # mask: [bs x f x c]  
        mask[index_tuple] = True
        x_freq[~mask] = torch.tensor(0.0+0j, device=x_freq.device)
        # ifft
        x = torch.fft.irfft(x_freq, dim=1, n=t)
        # mlp
        logits = self.fc(x)                                     # logits: [bs x n_query]
        # gumbel softmax
        q_indices = F.gumbel_softmax(logits, tau=1, hard=True)  # q_indices: [bs x n_query]

        return q_indices

 
class Extractor(nn.Module):
    def __init__(self, layers, 
                 context_size=64, query_len=5, d_model=128, decay=0.99, epsilon=1e-5
                ):
        super().__init__()
        # context
        self.context_size = context_size
        self.query_len = query_len
        self.d_model = d_model
        self.register_buffer("context", torch.randn(context_size, query_len, d_model))                  # context: [N x query_len x d_model]
        self.register_buffer("ema_count", torch.ones(context_size))   
        self.register_buffer("ema_dw", torch.zeros(context_size, query_len, d_model))                
        self.decay = decay
        self.epsilon = epsilon
        # extractor
        self.extractor = nn.ModuleList(layers)
              
    def update_context(self, q):
        # q: [bs x query_len x d_model]

        _, q_len, d = q.shape
        q_flat = q.reshape(-1, q_len*d)                                                                 # [bs x query_len*d_model]
        g_flat = self.context.reshape(-1, q_len*d)                                                      # [N x query_len*d_model]
        N, D = g_flat.shape

        distances = (
            torch.sum(q_flat**2, dim=1, keepdim=True) + 
            torch.sum(g_flat**2, dim=1) -
            2 * torch.matmul(q_flat, g_flat.t())
        )                                                                                               # [bs x N] soft
        # distances = torch.sum((q_flat.unsqueeze(1)-g_flat.unsqueeze(0))**2, dim=-1)                   
        indices = torch.argmin(distances.float(), dim=-1)                                               # [bs] 
        encodings = F.one_hot(indices, N).float()                                                       # [bs x N] hard
        q_context = torch.einsum("bn,nqd->bqd", [encodings, self.context])                              # [bs x query_len x d_model]
        q_hat = torch.einsum("bn,bqd->nqd", [encodings, q])                                             # [N x query_len x d_model]
        
        # query_latent_distances
        query_latent_distances = torch.mean(F.mse_loss(q_context.detach(), q, reduction="none"), dim=(1, 2))    # [bs]

        if self.training:
            with torch.no_grad():
                self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)   # [N]
                n = torch.sum(self.ema_count)
                self.ema_count = (self.ema_count + self.epsilon) / (n + D * self.epsilon) * n
                
                dw = torch.einsum("bn,bqd->nqd", [encodings, q])                                                # [N x query_len x d_model]
                self.ema_dw = self.decay * self.ema_dw + (1 - self.decay) * dw                                  # [N x query_len x d_model]
                self.context = self.ema_dw / self.ema_count.unsqueeze(-1).unsqueeze(-1)
        return query_latent_distances, q_hat
               
    def concat_context(self, context):
        return context.view(-1, self.d_model)                                                       # [N*query_len x d_model]

    def forward(self, q, local_repr, mask=None):
        # q: [bs, query_len, d_model]
        # local_repr: [bs, ms_t, d_model]
        for layer in self.extractor:
            q = layer(q, local_repr, mask)                                                          # [bs x query_len x d_model]

        query_latent_distances, q_hat = self.update_context(q) 
        context = self.concat_context(q_hat+self.context.detach()-q_hat.detach())                   # context: [N*query_len x d_model]
        return query_latent_distances, context                                                      # N,query_len,d - bs,q_query_len,d_model
        

class ExtractorLayer(nn.Module):
    def __init__(self, cross_attention, d_model, d_ff=None, norm="batchnorm", dropout=0.1, activation="relu"):
        super().__init__()

        d_ff = d_ff or 4 * d_model
        # attention
        self.cross_attention = cross_attention
        # ffn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # norm
        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, q, local_repr, mask=None):
        # q: [bs, query_len, d_model]
        # local_repr: [bs, ms_t, d_model]

        # cross_attention
        q = q + self.dropout(self.cross_attention(
            q, local_repr, local_repr,
            attn_mask=mask
        )[0])
        q = self.norm1(q)                                                       # q: [bs x query_len x d_model]

        # ffn
        y = self.dropout(self.activation(self.conv1(q.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))                        # y: [bs x query_len x d_model]

        return self.norm2(q + y)

#------------------- EnDec.py -------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x, x_attn_weights, cross_attn_weights = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, x_attn_weights, cross_attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, norm="batchnorm", dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # norm
        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm3 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x_, x_attn_weights = self.self_attention(
            x, x, x,
            attn_mask=x_mask,
        )
        x = x + self.dropout(x_)
        x = self.norm1(x)

        x_, cross_attn_weights = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
        )
        x = x + self.dropout(x_)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), x_attn_weights, cross_attn_weights
    

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, attn_weights = layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x, attn_weights
    

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, norm="batchnorm", dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # norm
        if "batch" in norm.lower():
            self.norm1 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm2 = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        x_, attn_weights = self.attention(
            x, x, x,
            attn_mask=attn_mask,
        )
        x = x + self.dropout(x_)
        x = self.norm1(x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn_weights

#--------------------------------- attention block -----------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=0.):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, attn_mask=None):
        bs, n_heads, q_len, d_k = q.size()
        scale = 1. / math.sqrt(d_k)
        attn_scores = torch.matmul(q, k) * scale                  # attn_scores: [bs x n_heads x q_len x k_len]
        if attn_mask is not None:                                 # attn_mask: [q_len x k_len]
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        attn_weights = F.softmax(attn_scores, dim=-1)             # attn_weights: [bs x n_heads x q_len x k_len]
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)                    # output: [bs x n_heads x q_len x d_v]
        
        return output.contiguous(), attn_weights
    

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_k=None, d_v=None, proj_dropout=0., qkv_bias=True):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.attn = attention

        self.proj = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q, K, V, attn_mask):
        bs, _, _ = Q.size()
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).permute(0,2,1,3)       # q_s: [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)       # k_s: [bs x n_heads x d_k x k_len]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).permute(0,2,1,3)       # v_s: [bs x n_heads x k_len x d_v]
        out, attn_weights = self.attn(q_s, k_s, v_s, attn_mask=attn_mask)             # out: [bs x n_heads x q_len x d_v], attn_weights: [bs x n_heads x q_len x k_len]
        out = out.permute(0,2,1,3).contiguous().view(bs, -1, self.n_heads * self.d_v) # out: [bs x q_len x n_heads * d_v]
        out = self.proj(out)                                                          # out: [bs x q_len x d_model]
        attn_weights = attn_weights.mean(dim=1)                                       # attn_weights: [bs x q_len x k_len]
        return out, attn_weights 


#----------------------------------- evaluator.py--------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from ts_ad_evaluation import f1
# from ts_ad_evaluation import affiliation
# from ts_ad_evaluation import auc_vus
# from ts_ad_evaluation import pate
# from ts_ad_evaluation import accomplish_UCR

import os


class Evaluator():
    def __init__(self, gt, anomaly_score, save_path):
        """
        input:
            gt: np.ndarray[int],
            anomaly_score: np.ndarray[float],
            save_path: str
        """
        self.gt = gt
        self.anomaly_score = anomaly_score
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def evaluate(self, metrics, merge=False, verbose=True, **metrics_args):
        """
        support metric: 'affiliation', 'auc', 'r_auc', 'vus', 'f1_raw', 'f1_pa', 'best_f1', 'f1_r', 'f1_event', 'pate', 'accomplish_UCR'

        input:
            metrics: List[str], e.g. ['affiliation', 'auc', 'r_auc', 'vus', 'f1_raw', 'f1_pa', 'best_f1', 'f1_r', 'f1_event', 'pate', 'accomplish_UCR']
            metrics_args: Dict[str, args], e.g. {'affiliation': [0.01, 0.02], 'f1_raw': [0.1, 0.2], ..., 'sliddingWindow': 100}
            merge: bool, if True: merge all results from different metrics
        output:
            results_storage: Dict[str, Dict[str, List[float]]]
        """
        results_storage = {}
        # metrics
        f1_evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        affiliation_evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        auc_vus_evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        # pate.evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        accomplish_evaluate(results_storage, metrics, labels=self.gt, score=self.anomaly_score, **metrics_args)
        # save
        self._save_csv(results_storage, metrics, merge=merge, verbose=verbose, **metrics_args)
        return results_storage
    
    def _save_csv(self, results_storage, metrics, merge, verbose, **metrics_args):
        if merge:
            df = pd.concat([pd.DataFrame.from_dict(results_storage[metric]) for metric in metrics], axis=1)
            if verbose: print(df)
            df.to_csv(f'{self.save_path}/_results.csv', index=0)
        else:
            for metric in metrics:
                df = pd.DataFrame.from_dict(results_storage[metric])
                if verbose: print(df)
                df.to_csv(f'{self.save_path}/_{metric}.csv', index=0)

    def find_thres(self, method, verbose=True, **args):
        """
        support method: 'prior_anomaly_rate', 'spot'  

        input:  
            method: str   
            args:  
            method == 'prior_anomaly_rate': require pAR: List[float]  
            method == 'spot': require init_score: np.ndarray[float]; q: List[float]  
        output:  
            thresholds: List[float]  
        
        example:  
            thres = evaluator.find_thres(method='prior_anomaly_rate', pAR=[0.05, 0.1])  
            thres = evaluator.find_thres(method='spot', init_score=init_score, q=[0.1, 0.2])  
        """
        if method == 'prior_anomaly_rate':
            thresholds = [np.percentile(self.anomaly_score, 100 * (1-pAR)) for pAR in args['pAR']]
            self._save_thres_info(args['pAR'], thresholds, method, verbose)
        elif method == 'spot':
            from .spot import SPOT
            thresholds = []
            for q in args['q']:
                s = SPOT(q)
                s.fit(args['init_score'], self.anomaly_score)
                s.initialize(verbose=False)
                ret = s.run()
                thresholds.append(np.mean(ret['thresholds']))
            self._save_thres_info(args['q'], thresholds, method, verbose)
            
        return thresholds
    
    def _save_thres_info(self, arg1, arg2, method, verbose):
        thres_info = pd.DataFrame(np.stack([arg1, arg2], axis=1), columns=['hyper-parameter', 'threshold'])
        if verbose: print(thres_info)
        thres_info.to_csv(f'{self.save_path}/_{method}_thres_info.csv', index=0)
    
    def vis_anomaly_intervals_all(self, series=None, start=None, end=None):
        plt.rcParams.update({'font.size': 14})
        
        if start is None: start = 0
        if end is None: end = len(self.gt)

        if series is None: nc = 1
        else: 
            nc = series.shape[-1]
            assert len(series) == len(self.anomaly_score)
            series = series[start:end]
            
        gt = self.gt[start:end]
        anomaly_score = self.anomaly_score[start:end]
        as_min, as_max = anomaly_score.min(0), anomaly_score.max(0)
        anomaly_score = (anomaly_score - as_min) / (as_max - as_min)

        borders = self._find_borders(gt)
        n_anomalies = len(borders)

        vis_path = os.path.join(self.save_path, "vis")
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
    
        for c in range(nc):
            fig, ax1 = plt.subplots(figsize=(20, 5))
            # series
            if series is not None:
                ax1.plot(range(start, end), series[:, c], linewidth=1, label='series', color='#5861AC')
                ax1.set_ylabel('Series value')
            # anomaly score
            ax2 = ax1.twinx()
            # ax2.plot(range(start, end), anomaly_score, linewidth=0.2, color='#72C3A3')
            ax2.fill_between(range(start, end), anomaly_score, 0, label='anomlay score', color='#72C3A3', alpha=0.3)
            ax2.set_ylabel('Anomaly score')
            ax2.set_xlabel('Time step')
            # abnormly interval
            for i in range(n_anomalies):
                if i == 0: 
                    plt.axvspan(xmin=borders[i][0]+start, xmax=borders[i][1]+start, ymin=0, ymax=1, label='anomaly interval', facecolor='r', alpha=0.3)
                else:
                    plt.axvspan(xmin=borders[i][0]+start, xmax=borders[i][1]+start, ymin=0, ymax=1, facecolor='r', alpha=0.2)
            
            fig.legend(loc='upper left', bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
            plt.savefig(f'{vis_path}/vis_{start}-{end}_c{c}.pdf', bbox_inches='tight')
            plt.clf()

    def vis_anomaly_intervals_each(self, series=None, max_span=100, max_anomalies=1):
        plt.rcParams.update({'font.size': 14})

        if series is None: nc = 1
        else: 
            nc = series.shape[-1]
            assert len(series) == len(self.anomaly_score)
            
        borders = self._find_borders(self.gt)
        n_anomalies = len(borders)

        if max_anomalies is None: anomalies_list = range(n_anomalies)
        elif isinstance(max_anomalies, int): anomalies_list = range(min(n_anomalies, max_anomalies))

        vis_path = os.path.join(self.save_path, "vis_anorm")
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        for c in range(nc):
            for i in anomalies_list:
                start = 0
                end = len(self.anomaly_score)
                if i:
                    start = borders[i-1][1]
                if i+1 < n_anomalies:
                    end = borders[i+1][0]-1
                
                if max_span is not None:
                    start = max(start, borders[i][0]-max_span)
                    end = min(end, borders[i][1]+max_span)

                fig, ax1 = plt.subplots(figsize=(6, 5))
                # series
                if series is not None:
                    series_i = series[start:end, c]
                    ax1.plot(range(start, end), series_i, linewidth=1, label='series', color='#5861AC')
                    ax1.set_ylabel('Series value')
                # anomaly score
                ax2 = ax1.twinx()
                score_i = self.anomaly_score[start:end]
                as_min, as_max = score_i.min(0), score_i.max(0)
                score_i = (score_i - as_min) / (as_max - as_min)
                # ax2.plot(range(start, end), score_i, linewidth=0.2, color='#72C3A3')
                ax2.fill_between(range(start, end), score_i, 0, label='anomlay score', color='#72C3A3', alpha=0.3)
                ax2.set_ylabel('Anomaly score')
                ax2.set_xlabel('Time step')

                # abnormly interval
                plt.axvspan(xmin=borders[i][0], xmax=borders[i][1], ymin=0, ymax=1, label='anomaly interval', facecolor='r', alpha=0.2)

                fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
                plt.savefig(f'{vis_path}/anorm{i}_{borders[i][0]}-{borders[i][1]}_c{c}.pdf', bbox_inches='tight')
                plt.clf()
                
    def _find_borders(self, gt):
        """
        return anomaly intervals: [[start1, end1)...[startn, endn)]
        """
        borders = []
        s = 0
        while True:
            if gt[s]==1:
                e = s
                while gt[e]==1:
                    if e == len(gt): break
                    e += 1
                borders.append([s, e])
                s = e+1
            else:
                s += 1
            if s >= len(gt): break
        return borders

#----------------------------- spot.py-----------------------------------------

from math import log, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'


class SPOT:
    def __init__(self, q=1e-4):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, verbose=True):
        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]

        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        zeros = np.concatenate((left_zeros, right_zeros))

        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True):
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        th = []
        alarm = []
        for i in tqdm(range(self.data.size)):

            if self.data[i] > self.extreme_quantile:
                if with_alarm:
                    alarm.append(i)
                else:
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)

            elif self.data[i] > self.init_threshold:
                self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                self.Nt += 1
                self.n += 1

                g, s, l = self._grimshaw()
                self.extreme_quantile = self._quantile(g, s)
            else:
                self.n += 1

            th.append(self.extreme_quantile)

        return {'thresholds': th, 'alarms': alarm}

    def plot(self, run_results, with_alarm=True):
        x = range(self.data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if 'thresholds' in K:
            th = run_results['thresholds']
            th_fig, = plt.plot(x, th, color=deep_saffron, lw=2, ls='dashed')
            fig.append(th_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            al_fig = plt.scatter(alarm, self.data[alarm], color='red')
            fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig

#----------------------------- data provider.py----------------------------------------------------------
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

def data_provider(root_path, datasets, batch_size, win_size=100, step=100, flag="train", percentage=1):
    if flag == "train": shuffle = True
    else: shuffle = False
    print(f"loading {datasets}({flag}) percentage: {percentage*100}% ...", end="")
    file_paths, train_lens = read_meta(root_path=root_path, dataset=datasets)
    discrete_channels = None
    if datasets == "MSL": discrete_channels = range(1, 55)
    if datasets == "SMAP": discrete_channels = range(1, 25)
    if datasets == "SWAT": discrete_channels =  [2,4,9,10,11,13,15,19,20,21,22,29,30,31,32,33,42,43,48,50]
    data_set = TrainSegLoader(file_paths, train_lens, win_size, step, flag, percentage, discrete_channels)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    print("done!")
    return data_set, data_loader

class TrainSegLoader(Dataset):
    def __init__(self, data_path, train_length, win_size, step, flag="train", percentage=0.1, discrete_channels=None):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        # 1.read data
        data = read_data(data_path)
        # 2.train
        train_data = data.iloc[:train_length, :]
        train_data, train_label =  (
            train_data.loc[:, train_data.columns != "label"].to_numpy(),
            train_data.loc[:, ["label"]].to_numpy(),
        )
        # 3.test
        test_data = data.iloc[train_length:, :]
        test_data, test_label =  (
            test_data.loc[:, test_data.columns != "label"].to_numpy(),
            test_data.loc[:, ["label"]].to_numpy(),
        )
        # 4.process
        if discrete_channels is not None:
            train_data = np.delete(train_data, discrete_channels, axis=-1)
            test_data = np.delete(test_data, discrete_channels, axis=-1)
        
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        if flag == "init":
            self.init = train_data
            self.init_label = train_label
        else:
            train_end = int(len(train_data) * 0.8)
            train_start = int(train_end*(1-percentage))
            self.train = train_data[train_start:train_end]
            self.train_label = train_label[train_start:train_end]
            self.val = train_data[train_end:]
            self.val_label = train_label[train_end:]
            self.test = test_data
            self.test_label = test_label

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "init":
            return (self.init.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index, eps=1):
        index = index * self.step
        if self.flag == "train":           
            return np.float32(self.train[index: index + self.win_size]), np.float32(self.train_label[index: index + self.win_size])
        elif self.flag == "val":
            return np.float32(self.val[index: index + self.win_size]), np.float32(self.val_label[index: index + self.win_size])
        elif self.flag == "test":
            return np.float32(self.test[index: index + self.win_size]), np.float32(self.test_label[index: index + self.win_size])
        elif self.flag == "init":
            return np.float32(self.init[index: index + self.win_size]), np.float32(self.init_label[index: index + self.win_size])
        else:
            return np.float32(self.test[index // self.step * self.win_size: index// self.step * self.win_size+ self.win_size]), np.float32(self.test_label[index // self.step * self.win_size: index // self.step * self.win_size + self.win_size])

def read_data(path: str, nrows=None) -> pd.DataFrame:
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values
    all_points = data.shape[0]
    columns = data.columns
    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()
    is_univariate = n_points == all_points
    n_cols = all_points // n_points
    df = pd.DataFrame()
    cols_name = data["cols"].unique()
    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    else:
        df[cols_name[0]] = data.iloc[:, 0]
    if label_exists:
        last_col_name = df.columns[-1]
        df.rename(columns={last_col_name: "label"}, inplace=True)
    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]
    return df

def read_meta(root_path, dataset):
    # Check if DETECT_META.csv exists, if not use default values
    meta_path = root_path + "/DETECT_META.csv"
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        meta = meta.query(f'file_name.str.contains("{dataset}")', engine="python")
        file_name = meta.file_name.values[0]
        train_lens = meta.train_lens.values[0]
    else:
        # Default values if DETECT_META.csv doesn't exist
        file_name = f"{dataset}.csv"
        train_lens_dict = {
            'SMAP': 135183, 'SWAT': 495000, 'MSL': 58317, 
            'PSM': 132481, 'SMD': 708405, 'SWAN': 60000, 'GECCO': 69260
        }
        train_lens = train_lens_dict.get(dataset, 10000)
    
    # Read from same folder as script (root_path) instead of root_path/data/
    file_paths = os.path.join(root_path, file_name)
    return file_paths, train_lens


#----------------------------------------------run.py

import argparse
import torch
import torch.backends
import random
import numpy as np


if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='')

    # basic config
    parser.add_argument('--mode', type=str, required=True, default="train", help='status')
    parser.add_argument('--configs_path', type=str, default="./configs/", help='')
    parser.add_argument('--save_path', type=str, default='./test_results/', help='')

    # data
    parser.add_argument('--root_path', type=str, default='/workspace/datasets/anomaly_detect', help='root path of the data file')
    parser.add_argument('--data', type=str, default='MSL', help='dataset type')
    parser.add_argument('--data_origin', type=str, default='DADA', help='dataset type')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # anomaly detection
    parser.add_argument('--method', type=str, default='spot', help='')
    parser.add_argument('--t', type=float, nargs='+', default=[0.1], help='')
    parser.add_argument('--metrics', type=str, nargs='+', default=['best_f1', 'auc', 'r_auc', 'vus'], help='')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # Exp
    exp = Exp_Anomaly_Detection(args, id=0)
    if args.mode == "train":
        exp.train()
        exp.test()
    elif args.mode == "test":
        exp.test()
    elif args.mode == "evaluate":
        exp.evaluate_spot(t=args.t)
    else:
        exp.analysis()

    if args.gpu_type == 'mps':
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
   

















