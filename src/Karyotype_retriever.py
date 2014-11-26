__author__ = 'ank'

from csv import reader
from os import path
from itertools import combinations
import warnings
from itertools import izip
import copy

import numpy as np
from matplotlib import pyplot as plt
from chiffatools import hmm
from chiffatools. Linalg_routines import rm_nans
from scipy.stats import ttest_ind
import scipy.cluster.hierarchy as sch
from scipy.ndimage.filters import gaussian_filter1d as smooth_signal

from src.Karyotype_support import t_test_matrix, rolling_window, pull_breakpoints, generate_breakpoint_mask, inflate_support, \
    center_and_rebalance_tags, recompute_level


pth = 'U:\\ank\\2014\\BRU_GBO\\4th_gen'
fle = 'mmc2-karyotypes.csv'

#####################################################
# Reads the source file for the amplification markers
#####################################################
with open(path.join(pth, fle)) as src:
    rdr = reader(src)
    header = rdr.next()
    selector = [1] + range(4, len(header))
    header = np.array(header)[selector]
    locuses = []
    for row in rdr:
        row = [ 'nan' if elt == 'NA' else elt for elt in row]
        locuses.append(np.genfromtxt(np.array(row))[selector].astype(np.float64))

################################
# Recovers the chromosome limits
################################
locuses = np.array(locuses)
chr_arr = locuses[:, 0]
chr_brps = pull_breakpoints(chr_arr)
#############################################################################################
# TODO: reformat so that instead of the brokenTable we have a set of chromosome breakpoints
broken_table = []
chroms = range(int(np.min(chr_arr)), int(np.max(chr_arr))+1)
for i in chroms:
    broken_table.append(chr_arr == i)
broken_table = np.array(broken_table)
#############################################################################################
chromosome_tag = np.repeat(locuses[:, 0].reshape((1, locuses.shape[0])), 200, axis=0)

#########################
# Defines the parsing HMM
#########################
transition_probs  = np.ones((3, 3)) * 0.001
np.fill_diagonal(transition_probs, 0.998)
initial_dist = np.array([[0.33, 0.34, 0.33]])
emission_probs  = np.ones((3, 3)) * 0.1
np.fill_diagonal(emission_probs, 0.8)
parsing_hmm = hmm.HMM(transition_probs, emission_probs)

warnings.catch_warnings()
warnings.simplefilter("ignore")


def simple_t_test_matrix(current_lane):
    dst = broken_table.shape[0]
    p_vals = np.empty((dst, dst))
    p_vals.fill(np.NaN)
    for i, j in combinations(range(0, dst), 2):
        _, p_val = ttest_ind(rm_nans(current_lane[broken_table[i, :]]), rm_nans(current_lane[broken_table[j, :]]), False)
        p_vals[i, j] = p_val
    return p_vals


def t_statistic_sorter(current_lane, breakpoints=None):

    if breakpoints is None:
        t_mat = simple_t_test_matrix(current_lane)
    else:
        t_mat = t_test_matrix(current_lane, breakpoints)

    t_mat[np.isnan(t_mat)] = 0
    t_mat = t_mat + t_mat.T
    np.fill_diagonal(t_mat, 1)
    ct_mat = t_mat.copy()
    ct_mat[t_mat < 0.01] = 0.01
    ct_mat = 1 - ct_mat

    Y = sch.linkage(ct_mat, method='centroid')
    clust_alloc = sch.fcluster(Y, 0.95, criterion='distance')

    averages = []
    if breakpoints is None:
        for i in range(0, 24):
            averages.append(np.median(rm_nans(current_lane[broken_table[i]])))
    else:
        subsets = np.split(current_lane, breakpoints[:-1])
        for subset in subsets:
            av = np.average(rm_nans(subset))
            averages.append(av)

    accumulator = [[] for _ in range(0, max(clust_alloc)+1)]
    for loc, item in enumerate(averages):
        accumulator[clust_alloc[loc]].append(item)

    accumulator = np.array([ np.average(np.array(_list)) for _list in accumulator][1:])

    splitting_matrix = np.repeat(accumulator.reshape((1, accumulator.shape[0])),
                                    accumulator.shape[0], axis = 0)
    splitting_matrix = np.abs(splitting_matrix - splitting_matrix.T)

    ########################################################################
    # plt.imshow(splitting_matrix, interpolation='nearest', cmap='coolwarm')
    # plt.show()
    ########################################################################

    Y_2 = sch.linkage(splitting_matrix, method='centroid')
    if breakpoints is None:
        clust_alloc_2 = sch.fcluster(Y_2, 3, criterion='maxclust')
    else:
        clust_alloc_2 = sch.fcluster(Y_2, 0.95, criterion='distance')  # attention, there is behavior-critical constant here

    accumulator_2 = [[] for _ in range(0, max(clust_alloc_2)+1)]
    for loc, item in enumerate(accumulator):
        accumulator_2[clust_alloc_2[loc]].append(item)
    accumulator_2 = np.array([ np.average(np.array(_list)) for _list in accumulator_2][1:])

    sorter_l = np.argsort(accumulator_2)
    sorter = dict((pos, i) for i, pos in enumerate(sorter_l))

    if breakpoints is None:
        re_chromosome_pad = np.repeat(chr_arr.reshape((1, chr_arr.shape[0])), 100, axis=0)
    else:
        pre_array = generate_breakpoint_mask(breakpoints)
        re_chromosome_pad = np.repeat(pre_array.reshape((1, pre_array.shape[0])), 100, axis=0)
        re_chromosome_pad += 1

    re_classification_tag = np.zeros(re_chromosome_pad.shape)

    for i in range(0, len(clust_alloc)):
        # re_classification_tag[re_chromosome_pad == i+1] = averages[i]
        # re_classification_tag[re_chromosome_pad == i+1] = accumulator[ clust_alloc[i]-1]
        re_classification_tag[re_chromosome_pad == i+1] = sorter[clust_alloc_2[clust_alloc[i]-1]-1]

    if breakpoints:
        re_classification_tag = center_and_rebalance_tags(re_classification_tag)

    #################################################################################
    # ax1 = plt.subplot(211)
    # plt.imshow(chromosome_tag, interpolation='nearest', cmap='spectral')
    # plt.imshow(re_classification_tag, interpolation='nearest', cmap='coolwarm')
    # plt.setp(ax1.get_xticklabels(), fontsize=6)
    # ax2 = plt.subplot(212, sharex=ax1)
    # plt.plot(current_lane-np.average(rm_nans(current_lane)), 'k.')
    # plt.setp(ax2.get_xticklabels(), visible=False)
    # plt.show()
    ################################################################################

    return re_classification_tag


def compute_karyotype(current_lane, plotting=False):

    def support_function(x, y):
        if x == 0 and y == 0:
            return 0
        if x == -1 and y <= 0:
            return -1
        if x >= 0 and y == 1:
            return 1
        if x == -1 and y == 1:
            return 0

    def plot_classification():

        classification_tag = np.repeat(parsed.reshape((1, parsed.shape[0])), 100, axis=0)
        ax1 = plt.subplot(311)
        plt.imshow(chromosome_tag, interpolation='nearest', cmap='spectral')
        plt.imshow(classification_tag, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=2)
        plt.setp(ax1.get_xticklabels(), fontsize=6)
        ax2 = plt.subplot(312, sharex=ax1)
        plt.plot(current_lane, 'k.')
        plt.plot(gauss_convolve, 'r', lw=2)
        plt.plot(gauss_convolve+rolling_std, 'g', lw=1)
        plt.plot(gauss_convolve-rolling_std, 'g', lw=1)
        plt.plot(segment_averages, 'b', lw=2)
        plt.axhline(y=threshold, color='c')
        plt.axhline(y=-threshold, color='c')
        # for point in breakpoints:
        #     plt.axvline(x=point, color='b')
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax3 = plt.subplot(313, sharex=ax1)
        plt.plot(current_lane-segment_averages, 'k.')
        # # plt.plot(total_corr, 'k.')
        # # plt.plot(gauss_convolve, 'k.')
        # # plt.plot(rolling_std, 'g')
        # plt.setp(ax3.get_xticklabels(), visible=False)
        plt.show()

    current_lane = current_lane - np.mean(rm_nans(current_lane))

    gauss_convolve = np.empty(current_lane.shape)
    gauss_convolve.fill(np.NaN)
    gauss_convolve[np.logical_not(np.isnan(current_lane))] = smooth_signal(rm_nans(current_lane), 10, order=0, mode='mirror')

    rolling_std = np.empty(current_lane.shape)
    rolling_std.fill(np.NaN)
    payload = np.std(rolling_window(rm_nans(current_lane), 10), 1)
    c1, c2 = (np.sum(np.isnan(current_lane[:5])), np.sum(np.isnan(current_lane[-4:])))
    rolling_std[5:-4][np.logical_not(np.isnan(current_lane))[5:-4]] = np.lib.pad(payload,
                                                                                 (c1, c2),
                                                                                 'constant',
                                                                                 constant_values=(np.NaN, np.NaN))

    corrfact = np.random.randint(-5, 5)
    threshold = np.percentile(rm_nans(rolling_std), 75+corrfact)
    binarized = (current_lane > threshold).astype(np.int16) - (current_lane < -threshold) + 1
    parsed = np.array(hmm.viterbi(parsing_hmm, initial_dist, binarized))

    breakpoints = pull_breakpoints(parsed)

    segment_averages = np.empty(current_lane.shape)
    subsets = np.split(current_lane, breakpoints)

    breakpoints.append(current_lane.shape[0])

    pre_brp = 0
    for subset, brp in izip(subsets, breakpoints):
        av = np.average(rm_nans(subset))
        segment_averages[pre_brp : brp] = av
        pre_brp = brp

    collector = []
    for i in chroms:
        lw = np.percentile(parsed[broken_table[i-1, :]]-1, 25)
        hr = np.percentile(parsed[broken_table[i-1, :]]-1, 75)
        collector.append(support_function(lw, hr))

    if plotting:
        plot_classification()

    return current_lane-segment_averages, segment_averages, parsed, np.array(collector)


def compute_recursive_karyotype(lane, plotting=False, debug_plotting=False):

    def support_function(x, y):
        if x == 0 and y == 0:
            return 0
        if x < 0 and y <= 0:
            return -1
        if x >= 0 and y > 0:
            return 1
        if x < 0 and y > 0:
            return 0

    def determine_locality():
        breakpoint_accumulator = []
        ###########################################################
        # repeated code. TODO: in the future, factor it out
        prv_brp = 0
        for breakpoint in breakpoints:
            breakpoint_accumulator.append(breakpoint-prv_brp)
            prv_brp = breakpoint
        breakpoint_accumulator = np.array(breakpoint_accumulator)
        ############################################################
        msk = breakpoint_accumulator < 25
        shortness = inflate_support(current_lane.shape[0], breakpoints, msk)

        shortness_breakpoints = pull_breakpoints(shortness[0, :])
        shortness_breakpoints.append(shortness.shape[1])

        chr_brp_arr = np.array(chr_brps)
        re_brp = []
        prv_brp = 0
        for breakpoint in shortness_breakpoints:
            chr_break_verify = np.logical_and(chr_brp_arr > prv_brp, chr_brp_arr < breakpoint)
            if any(chr_break_verify) and all(shortness[0, :][prv_brp:breakpoint]):
                re_brp += chr_brp_arr[chr_break_verify].tolist()
            prv_brp = breakpoint
        shortness_breakpoints = sorted(shortness_breakpoints + re_brp)

        shortness_ladder = inflate_support(current_lane.shape[0], shortness_breakpoints)[0, :]

        filled_in = re_class_tag.copy().astype(np.float)
        levels = amplicons.copy()
        prv_brp = 0
        processing_trace = []
        non_short_selector = np.logical_not(shortness[0, :].astype(np.bool))
        for _i, breakpoint in enumerate(shortness_breakpoints):
            processing_trace.append(1)
            if all(shortness[0, :][prv_brp:breakpoint]):
                processing_trace = processing_trace[:-1]
                current_fltr = shortness_ladder == _i
                diff_min_max = np.max(parsed[current_fltr]) - np.min(parsed[current_fltr])
                if diff_min_max == 0 and prv_brp - 1 > 0 and breakpoint + 1 < shortness_breakpoints[-1]:
                    processing_trace.append(0)
                    # print parsed[prv_brp-1] == parsed[breakpoint+1], breakpoint in chr_brps, prv_brp in chr_brps,
                    non_modified = True
                    if parsed[prv_brp-1] == parsed[breakpoint+1]:
                        processing_trace[-1] -= 0.1
                        filled_in[:, current_fltr] = parsed[prv_brp - 1]
                        levels[current_fltr] = amplicons[prv_brp - 1]
                        non_modified = False
                    if prv_brp in chr_brps and breakpoint not in chr_brps:
                        processing_trace[-1] -= 0.2
                        filled_in[:, current_fltr] = parsed[breakpoint + 1]
                        levels[current_fltr] = amplicons[breakpoint + 1]
                        non_modified = False
                    if breakpoint in chr_brps and prv_brp not in chr_brps:
                        processing_trace[-1] -= 0.3
                        filled_in[:, current_fltr] = parsed[prv_brp - 1]
                        levels[current_fltr] = amplicons[prv_brp - 1]
                        non_modified = False
                    if non_modified:
                        processing_trace[-1] -= 0.4
                        cur_chr = chr_arr[breakpoint]
                        chr_median = np.median(amplicons[np.logical_and(cur_chr, non_short_selector)])
                        lar = np.array([prv_brp-1, breakpoint+1])
                        lval = amplicons[lar]
                        vl = np.argmin(lval-chr_median)
                        filled_in[:, current_fltr] = parsed[lar[vl]]
                        levels[current_fltr] = amplicons[lar[vl]]
                else :
                    processing_trace.append(2)
                    average = np.average(amplicons[current_fltr])
                    closest_index = np.argmin(np.abs(amplicons[non_short_selector] - average))
                    closest_index = np.array(range(0, shortness_breakpoints[-1]))[non_short_selector][closest_index]
                    color = parsed[closest_index]
                    filled_in[:, current_fltr] = color
                    levels[current_fltr] = amplicons[closest_index]
            prv_brp = breakpoint

        filled_in = center_and_rebalance_tags(filled_in)
        levels = recompute_level(filled_in[0, :], levels)
        # print 'execution trace:', processing_trace

        #######################################################################################
        # ax1 = plt.subplot(411)
        # plt.imshow(chromosome_tag, interpolation='nearest', cmap='spectral')
        # plt.imshow(re_class_tag, interpolation='nearest', cmap='coolwarm')
        # plt.setp(ax1.get_xticklabels(), fontsize=6)
        # ax2 = plt.subplot(412, sharex=ax1)
        # plt.plot(locuses[:, lane] - np.average(rm_nans(locuses[:, lane])), 'k.')
        # show_breakpoints(shortness_breakpoints)
        # plt.plot(amplicons, 'r', lw=2)
        # plt.plot(levels, 'g', lw=2)
        # plt.setp(ax2.get_xticklabels(), visible=False)
        # ax3 = plt.subplot(413, sharex=ax1)
        # plt.imshow(chromosome_tag, interpolation='nearest', cmap='spectral')
        # plt.imshow(inflate_tags(shortness_ladder), interpolation='nearest', cmap='coolwarm')
        # plt.setp(ax3.get_xticklabels(), visible=False)
        # ax4 = plt.subplot(414, sharex=ax1)
        # plt.imshow(chromosome_tag, interpolation='nearest', cmap='spectral')
        # plt.imshow(inflate_support(current_lane.shape[0], shortness_breakpoints, np.array(processing_trace)), interpolation='nearest', cmap='coolwarm')
        # plt.setp(ax4.get_xticklabels(), visible=False)
        # plt.show()
        #######################################################################################

        return  shortness, filled_in, levels


    current_lane = locuses[:, lane]
    retlist = compute_karyotype(current_lane, plotting=debug_plotting)

    amplicons = retlist[1]
    ampli_levels = retlist[2]-1
    re_retlist = copy.deepcopy(retlist)
    for i in range(0, 6):
        re_retlist = compute_karyotype(re_retlist[0], plotting=debug_plotting)
        if np.max(re_retlist[2])-np.min(re_retlist[2]) < 1:
            break
        else:
            amplicons += re_retlist[1]
            ampli_levels += re_retlist[2] - 1

    breakpoints = pull_breakpoints(ampli_levels)
    breakpoints.append(current_lane.shape[0])

    re_class_tag = t_statistic_sorter(current_lane, breakpoints)
    parsed = re_class_tag[0, :]
    local, background, corrected_levels = determine_locality()

    parsed = background[0, :]
    collector = []
    for i in chroms:
        lw = np.percentile(parsed[broken_table[i-1, :]], 25)
        hr = np.percentile(parsed[broken_table[i-1, :]], 75)
        collector.append(support_function(lw, hr))

    if plotting:
        ax1 = plt.subplot(511)
        plt.imshow(chromosome_tag, interpolation='nearest', cmap='spectral')
        plt.imshow(background, interpolation='nearest', cmap='coolwarm')
        # plt.imshow(re_class_tag, interpolation='nearest', cmap='coolwarm')
        plt.setp(ax1.get_xticklabels(), fontsize=6)
        ax2 = plt.subplot(512, sharex=ax1)
        plt.plot(locuses[:, lane]-np.average(rm_nans(locuses[:, lane])), 'k.')
        plt.plot(amplicons, 'r', lw=2)
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax3 = plt.subplot(513, sharex=ax1, sharey=ax2)
        # plt.plot(locuses[:, lane]-np.average(rm_nans(locuses[:, lane])), 'k.')
        plt.plot(corrected_levels, 'r', lw=2)
        plt.setp(ax3.get_xticklabels(), visible=False)
        ax4 = plt.subplot(514, sharex=ax1, sharey=ax2)
        # plt.plot(locuses[:, lane]-np.average(rm_nans(locuses[:, lane])), 'k.')
        plt.plot(amplicons-corrected_levels, 'g', lw=2)
        plt.setp(ax4.get_xticklabels(), visible=False)
        ax5 = plt.subplot(515, sharex=ax1)
        # plt.plot(locuses[:, lane]-np.average(rm_nans(locuses[:, lane])), 'k.')
        # plt.plot(corrected_levels, 'r', lw=2)
        # plt.plot(amplicons-corrected_levels, 'g', lw=2)
        plt.setp(ax5.get_xticklabels(), visible=False)
        # ax6 = plt.subplot(616, sharex=ax1)
        plt.imshow(chromosome_tag, interpolation='nearest', cmap='spectral')
        plt.imshow(inflate_support(chromosome_tag.shape[1], chr_brps, np.array(collector)), interpolation='nearest', cmap='coolwarm', vmin=-1., vmax=1 )
        # plt.setp(ax6.get_xticklabels(), visible=False)
        plt.show()

    return collector


def compute_all_karyotypes():

    def plot():
        plt.imshow(chromlist, interpolation='nearest', cmap='coolwarm')
        plt.show()

    chromlist = []
    for i in range(1, locuses.shape[1]):
        chromlist.append(compute_recursive_karyotype(i, plotting=False))
    chromlist = np.array(chromlist).astype(np.float64)
    plot()
    return chromlist, header[1:locuses.shape[1]]


if __name__ == "__main__":
    # compute_karyotype(6, True, 0.35)
    print compute_all_karyotypes()