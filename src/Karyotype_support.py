__author__ = 'ank'

import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
from chiffatools. Linalg_routines import rm_nans
from scipy.stats import ttest_ind


def t_test_matrix(current_lane, breakpoints):
    dst = len(breakpoints)
    p_vals = np.empty((dst, dst))
    p_vals.fill(np.NaN)
    subsets = np.split(current_lane, breakpoints[:-1])
    for i, j in combinations(range(0, dst), 2):
        _, p_val = ttest_ind(rm_nans(subsets[i]), rm_nans(subsets[j]), False)
        p_vals[i, j] = p_val
    return p_vals


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def pull_breakpoints(contingency_list):
    no_nans_parsed = rm_nans(contingency_list)
    contingency = np.lib.pad(no_nans_parsed[:-1] == no_nans_parsed[1:], (1, 0), 'constant', constant_values=(True, True))
    nans_contingency = np.zeros(contingency_list.shape).astype(np.bool)
    nans_contingency[np.logical_not(np.isnan(contingency_list))] = contingency
    breakpoints = np.nonzero(np.logical_not(nans_contingency))[0].tolist()
    return breakpoints


def show_breakpoints(breakpoints):
    for point in breakpoints:
        plt.axvline(x=point, color='b')


def generate_breakpoint_mask(breakpoints):
    support = np.zeros((np.max(breakpoints), ))
    pre_brp = 0
    for i, brp in enumerate(breakpoints):
        support[pre_brp:brp] = i
        pre_brp = brp
    return support


def inflate_support(length, breakpoints, values=None):
    if values is None:
        values = np.array(range(0, len(breakpoints)))
    if breakpoints[-1]< length:
        breakpoints.append(length)
    ret_array = np.zeros((100, length))
    for _i in range(1, values.shape[0]):
        ret_array[:, breakpoints[_i-1]: breakpoints[_i]] = values[_i]
    return ret_array


def inflate_tags(_1D_array, width=100):
    nar = _1D_array.reshape((1, _1D_array.shape[0]))
    return np.repeat(nar, width, axis=0)


def center_and_rebalance_tags(source_array):
    """
    Attention, this method is susceptible to create wrong level distribution in case less than 33 percent of the
    genome is in the base ploidy state.

    :param source_array:
    :return:
    """

    def correct_index(mp_vals):
        if len(lvls)>7:
            med_min = np.percentile(source_array, 34)
            med_max = np.percentile(source_array, 66)
            med_med = np.median(source_array)
            lcm_med = [mp_vals[_i] for _i, _val in enumerate(lvls) if _val == med_med][0]
            for _i, _lvl in enumerate(lvls):
                if _lvl >= med_min and _lvl <= med_max:
                    mp_vals[_i] = lcm_med
        return mp_vals

    lvls = np.unique(source_array).tolist()
    map_values = correct_index(np.array(range(0, len(lvls))).astype(np.float))
    index = np.digitize(source_array.reshape( -1, ), lvls) - 1
    source_array = map_values[index].reshape(source_array.shape)
    arr_med = np.median(source_array)
    arr_min = np.min(source_array)
    arr_max = np.max(source_array)
    source_array -= arr_med
    source_array[source_array < 0] /= (arr_med - arr_min)
    source_array[source_array > 0] /= (arr_max - arr_med)
    return source_array


def recompute_level(labels, mean_values):
    new_mean_values = np.zeros(mean_values.shape)
    lvls = np.unique(labels).tolist()
    for _value in lvls:
        current_mask = labels == _value
        new_mean_values[current_mask] = np.average(mean_values[current_mask])
    return  new_mean_values