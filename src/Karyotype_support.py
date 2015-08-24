__author__ = 'ank'

import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
from chiffatools.Linalg_routines import rm_nans
from chiffatools import hmm
from scipy.stats import ttest_ind
import scipy.cluster.hierarchy as sch
from scipy.ndimage.filters import gaussian_filter1d
from pprint import pprint
from chiffatools.dataviz import smooth_histogram
from basic_drawing import show_2d_array
from scipy.stats import norm, poisson


def support_function(x, y):
    if x == 0 and y == 0:
        return 0
    if x == -1 and y <= 0:
        return -1
    if x >= 0 and y == 1:
        return 1
    if x == -1 and y == 1:
        return 0


def brp_retriever(array, breakpoints_set):
    """
    Retrieves values on the segments defined by the breakpoints.

    :param array:
    :param breakpoints_set:
    :return: the values in array contained between breakpoints
    """
    breakpoints_set = sorted(list(set(breakpoints_set))) # sorts tbe breakpoints
    if breakpoints_set[-1] == array.shape[0]:
        breakpoints_set = breakpoints_set[:-1]  # just in case end of array was already included
    values = np.split(array, breakpoints_set) # retrieves the values
    return values


def brp_setter(breakpoints_set, prebreakpoint_values):
    """
    Creates an array of the size defined by the biggest element of the breakpoints set and sets the
    intervals values to prebreakpoint_values. It assumes that the largest element of breakpoints set
    is equal to the size of desired array

    :param array:
    :param breakpoints_set:
    :param prebreakpoint_values:
    :return:
    """
    assert(len(breakpoints_set)==len(prebreakpoint_values))
    breakpoints_set = sorted(list(set(breakpoints_set))) # sorts the breakpoints
    support = np.empty((breakpoints_set[-1], )) # creates array to be filled
    support.fill(np.nan)

    pre_brp = 0 # fills the array
    for value, brp in zip(prebreakpoint_values, breakpoints_set):
        support[pre_brp:brp] = value
        pre_brp = brp

    return support


def HMM_constructor(coherence_length):
    """
    Builds an HMM with the defined coherence length (minimal number of deviating points before a switch occurs
    to a different level)

    :param coherence_length:
    :return:
    """

    el2 = 0.1
    el1 = el2/np.power(10, coherence_length/2.0)

    # print 'el1: %s, el2: %s'% (el1, el2)

    transition_probs = np.ones((3, 3)) * el1
    np.fill_diagonal(transition_probs, 1 - 2*el1)

    emission_probs = np.ones((3, 3)) * el2
    np.fill_diagonal(emission_probs, 1 - 2*el2)

    return hmm.HMM(transition_probs, emission_probs)


def t_test_matrix(samples, jacknife_size = 50):
    """
    Performs series of t_tests between the segments of current lane divided by breakpoints. In addition to
    that, uses an inner re-sampling to prevent discrepancies due to lane size

    Alternative: use normalized differences in level (mean/std)
    Even better: use Tuckey test

    :param current_lane: fused list of chromosomes
    :param breakpoints: list of positions where HMM detected significant differences between elements
    :param jacknife_size: if None or 0, will run T-test on the entire sample. Otherwise will use the provided integert
                            to know how many elements from each collection to sample for a t-test.
    :return: matrix of P values student's t_test of difference between average ploidy segments
    """
    def inner_prepare(array):
        if jacknife_size:
            return rm_nans(np.random.choice(array,
                                        size = (jacknife_size,),
                                        replace=True))
        else:
            return rm_nans(array)

    samples_number = len(samples)
    p_vals_matrix = np.empty((samples_number, samples_number))
    p_vals_matrix.fill(np.NaN)
    for i, j in combinations(range(0, samples_number), 2):
        _, p_val = ttest_ind(inner_prepare(samples[i]), inner_prepare(samples[j]), False)
        p_vals_matrix[i, j] = p_val
    return p_vals_matrix


def t_test_collapse(set_of_samples):
    """
    Takes in a set of samples and returns per-sample means and the groups of means that are statistically not different

    :param set_of_samples:
    :return:
    """
    print 'set of samples length:', len(set_of_samples)

    nanmeans = [np.nanmean(x) for x in set_of_samples]

    t_mat = t_test_matrix(set_of_samples, None)   # generate T-test matrix

    t_mat[np.isnan(t_mat)] = 0
    t_mat = t_mat + t_mat.T
    np.fill_diagonal(t_mat, 1)
    ct_mat = t_mat.copy()
    ct_mat[t_mat < 0.01] = 0.01
    ct_mat = 1 - ct_mat         # transform into a reasonable distance matrix

    show_2d_array(ct_mat, 'Inverted p_value matrix')

    Y = sch.linkage(ct_mat, method='centroid')
    clust_alloc = sch.fcluster(Y, 0.95, criterion='distance')-1  # merge on the 5% rule

    # groups together elements that are not statistcally significantly different at the 5% level
    accumulator = [[] for _ in range(0, max(clust_alloc)+1)]
    for loc, item in enumerate(nanmeans):
        accumulator[clust_alloc[loc]].append(item)

    accumulator = np.array([ np.nanmean(np.array(_list)) for _list in accumulator])

    collapsed_means = np.empty_like(nanmeans)
    collapsed_means.fill(np.nan)

    for i, j in enumerate(clust_alloc.tolist()):
        collapsed_means[i] = accumulator[j]

    return nanmeans, collapsed_means


def collapse_means(set_of_means, set_of_stds):
    pass


def Tukey_outliers(set_of_means, FDR=0.005, supporting_interval=0.5, verbose=False):
    """
    Performs Tukey quintile test for outliers from a normal distribution with defined false discovery rate

    :param set_of_means:
    :param FDR:
    :return:
    """
    # false discovery rate v.s. expected falses v.s. power
    q1_q3 = norm.interval(supporting_interval)
    FDR_q1_q3 = norm.interval(1 - FDR)
    multiplier = (FDR_q1_q3[1] - q1_q3[1]) / (q1_q3[1] - q1_q3[0])
    l_means = len(set_of_means)

    if verbose:
        print "FDR: %s %%, expected outliers: %s, outlier 5%% confidence interval: %s"% (FDR*100, FDR*l_means,
                                                                                  poisson.interval(0.95, FDR*l_means))

    q1 = np.percentile(set_of_means, 50*(1-supporting_interval))
    q3 = np.percentile(set_of_means, 50*(1+supporting_interval))
    high_fence = q1 + multiplier*(q3 - q1)
    low_fence = q3 - multiplier*(q3 - q1)

    ho = (set_of_means < low_fence).nonzero()[0]
    lo = (set_of_means > high_fence).nonzero()[0]

    return lo, ho


def binarize(current_lane, FDR=0.05):
    """
    Retrieves the outliers for the HMM to process

    :param current_lane: array of markers in order
    :param FDR: false discovery rate
    :return:
    """
    binarized = np.empty_like(current_lane)
    binarized.fill(1)

    lo, ho = Tukey_outliers(current_lane, FDR)

    binarized[ho] = 0
    binarized[lo] = 2 # Inverted because of legacy reasons

    return binarized


def old_padded_means(lane, HMM_levels):
    """
    padding of each contig separately

    :param lane:
    :param HMM_levels:
    :return:
    """
    breakpoints = pull_breakpoints(HMM_levels)
    breakpoints = sorted(list(set(breakpoints + [lane.shape[0]])))
    averages = [np.nanmean(x) for x in brp_retriever(lane, breakpoints)]
    return  brp_setter(breakpoints, averages)


def padded_means(lane, HMM_decisions):
    """
    Computes means of each HMM-detected layer separately

    :param lane:
    :param HMM_decisions:
    :return:
    """
    return_array = np.empty_like(lane)
    return_array[HMM_decisions == -1] = np.nanmean(lane[HMM_decisions == -1])
    return_array[HMM_decisions == 1] = np.nanmean(lane[HMM_decisions == 1])
    return_array[HMM_decisions == 0] = np.nanmean(lane[HMM_decisions == 0])
    return return_array


def gini_coeff(x):
    """
    requires all values in x to be zero or positive numbers,
    otherwise results are undefined
    source : http://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html
    """
    x = np.abs(x)
    x = rm_nans(x.astype(np.float))
    n = len(x)
    s = x.sum()
    r =  np.argsort(np.argsort(-x)) # calculates zero-based ranks
    return 1 - (2.0 * (r*x).sum() + s)/(n*s)


def model_stats(regressed, regressor):
    r2_0 = np.nansum(np.power(regressed, 2))
    r2 = np.nansum(np.power(regressed-regressor, 2))

    to1 = np.empty_like(regressed)
    to1.fill(np.nan)
    to2 = np.empty_like(regressed)
    to2.fill(np.nan)

    FDR = 0.01
    tor = Tukey_outliers(regressed, FDR)
    tor2 = Tukey_outliers(regressed - regressor, FDR)
    tor2 = (np.array(list(set(tor2[0].tolist()).intersection(set(tor[0].tolist())))),
           np.array(list(set(tor2[1].tolist()).intersection(set(tor[1].tolist())))))

    to1[tor[0]] = regressed
    to1[tor[1]] = regressed
    to2[tor2[0]] = regressed - regressor
    to2[tor2[1]] = regressed - regressor

    complexity = (len(pull_breakpoints(regressor)))/2

    to1 = np.nansum(np.power(to1, 2))
    to2 = np.nansum(np.power(to2, 2))

    improvement_1 = (1-r2/r2_0)*100
    improvement_2 = (1-to2/to1)*100

    l1 = (tor[0].shape[0] + tor[1].shape[0])/FDR/regressed.shape[0] # since poisson sf fails for such big deviations
    l2 = (tor2[0].shape[0] + tor2[1].shape[0])/FDR/regressed.shape[0]

    return improvement_1, improvement_2, complexity, l1-l2, complexity-l1+l2


def model_decision(std_reduction, std_of_outliers, segments_No, times_No_outliers_reduced, surrogate_AIC):

    return (std_of_outliers > 10 or std_reduction > 10 or surrogate_AIC < 0) and times_No_outliers_reduced > 1

def gini_compression(set_of_means):
    set_of_means = rm_nans(set_of_means.astype(np.float))
    sorting_index = np.argsort(set_of_means)
    sorted_set = set_of_means[sorting_index]
    sorted_set -= np.min(sorted_set)
    total = sorted_set.sum()
    normalized_set = sorted_set/total
    csum = np.cumsum(normalized_set)
    gini = (np.linspace(0, 1, csum.shape[0])-csum).sum()/float(csum.shape[0])

    plt.title(gini)
    plt.plot(csum, 'k')
    plt.plot(np.linspace(0, 1, csum.shape[0]), 'r')
    plt.show()


def rolling_window(base_array, window_size):
    """
    Extracts a rolling subarray from the current array of a provided size

    :param base_array: array to which we want to apply the rolling window
    :param window_size: the size of the rolling window
    :return:
    """
    shape = base_array.shape[:-1] + (base_array.shape[-1] - window_size + 1, window_size)
    strides = base_array.strides + (base_array.strides[-1],)
    return np.lib.stride_tricks.as_strided(base_array, shape=shape, strides=strides)


def pull_breakpoints(contingency_list):
    """
    A method to extract breakpoints separating np.array regions with the same value.

    :param contingency_list: np.array containing regions of identical values
    :return: list of breakpoint indexes
    """
    no_nans_parsed = rm_nans(contingency_list)
    contingency = np.lib.pad(no_nans_parsed[:-1] == no_nans_parsed[1:], (1, 0), 'constant', constant_values=(True, True))
    nans_contingency = np.zeros(contingency_list.shape).astype(np.bool)
    nans_contingency[np.logical_not(np.isnan(contingency_list))] = contingency
    breakpoints = np.nonzero(np.logical_not(nans_contingency))[0].tolist()
    return breakpoints


def show_breakpoints(breakpoints, color = 'k'):
    """
    plots the breakpoints

    :param breakpoints:
    :return:
    """
    for point in breakpoints:
        plt.axvline(x=point, color=color)


def generate_breakpoint_mask(breakpoints):
    """
    generates mask assigning a different integer to each breakpoint

    :param breakpoints:
    :return:
    """
    support = np.zeros((np.max(breakpoints), ))
    pre_brp = 0
    for i, brp in enumerate(breakpoints):
        support[pre_brp:brp] = i
        pre_brp = brp
    return support


def inflate_support(length, breakpoints, values=None):
    """
    transforms 1D representation of chromosomes into a 2d array that can be rendered with an eventual filter on breakpoints

    :param length:
    :param breakpoints:
    :param values:
    :return:
    """

    if values is None:
        values = np.array(range(0, len(breakpoints)))
    if breakpoints[-1]< length:
        breakpoints.append(length)
    ret_array = np.zeros((100, length))
    for _i in range(1, values.shape[0]):
        ret_array[:, breakpoints[_i-1]: breakpoints[_i]] = values[_i]
    return ret_array


def inflate_tags(_1D_array, width=100):
    """
    reshapes a 1_d array into a 2d array that can be rendered

    :param _1D_array:
    :param width:
    :return:
    """
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
            lcm_med = [mp_vals[_i] for _i, _val in enumerate(lvls) if np.abs(_val - med_med) < 0.01][0]
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


def position_centromere_breakpoints(chr2centromere_loc, chr_location_arr, broken_table):
    """
    Extracts indexes where the centromeric breakpoints indexes are in the locus array

    :param chr2centromere_loc: dict mapping chrmosome number to cetromere location (in kb)
    :param chr_location_arr: array containing chromosome number inf 1st column and locus location on that chromosome
    in the second.
    :param broken_table: chromosome partition mask collection
    :return:
    """
    numerical_pad = np.arange(chr_location_arr.shape[0])[:, np.newaxis]
    chr_location_arr = np.hstack((numerical_pad, chr_location_arr))
    global_indexes = np.zeros((len(chr2centromere_loc.keys()),)).tolist()
    for chromosome, centromere_location in chr2centromere_loc.iteritems():
        local_idx = np.argmax(chr_location_arr[broken_table[chromosome-1], 2] > centromere_location)
        global_indexes[chromosome-1] = chr_location_arr[broken_table[chromosome-1], 0][local_idx]
    return global_indexes


if __name__ == "__main__":
    # print HMM_constructor(3)
    # print means_collapse(np.linspace(0,1,10), 0.1)
    # gini_compression(np.linspace(0,1,10))
    gini_compression(np.random.normal(size=100))
    gini_compression(np.random.poisson(5, 100))