__author__ = 'ank'

from csv import reader
from os import path
import warnings
from itertools import izip
import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import scipy.cluster.hierarchy as sch
from scipy.ndimage.filters import gaussian_filter1d as smooth_signal
from pprint import pprint
import sys
from chiffatools import hmm
from chiffatools.Linalg_routines import rm_nans
from src.pre_processors import get_centromeres
from src.Karyotype_support import t_test_matrix, rolling_window, pull_breakpoints, generate_breakpoint_mask, inflate_support, \
    center_and_rebalance_tags, recompute_level, show_breakpoints, position_centromere_breakpoints, inflate_tags, HMM_constructor
import src.Karyotype_support as KS
from src.Drawing_functions import plot_classification, multi_level_plot, show_2d_array

# TODO: resolve the weird interference between the min length and breakpoint collapse algorithms
# TODO: => invert collapse and separation algorithms?

# TODO: tor the T-test, in addition to the actual p_value, use actual size of difference (p_val for a sample of 10?)
# TODO: force chromosome breakpoints re-set before computing the pyramid of collapses

##################################################################################
# suppresses the noise from Numpy about future suppression of the boolean mask use
##################################################################################
# warnings.catch_warnings()
# warnings.simplefilter("ignore")

####################################################
# Defines the parsing HMM and matrices underlying it
####################################################
initial_dist = np.array([[0.33, 0.34, 0.33]])
parsing_hmm = HMM_constructor(10)


class Environement(object):
    """
    Just a payload object to carry around environment variables that are specific to the implementation
    """

    def __init__(self, _path, _file, min_seg_support=25, collapse_segs=3, debug_level=0):

        self.header = None
        self.chr_brps = None
        self.broken_table = None
        self.chromosome_tag = None
        self.centromere_brps = None
        self.locus_locations = None
        self.chr_arr = None
        self.chroms = None
        self.locuses = None
        self.segments = None
        self.min_seg_support = min_seg_support
        self.collapse_segs = collapse_segs
        self.debug_level = debug_level

        if _file is not None:
            paramdict = self.wrap_import(_path, _file)
            for name, value in paramdict.iteritems():
                setattr(self, name, value)


    def wrap_import(self, _path, _file):
        """
        Performs an overall import of values from the source file

        :param _path:
        :param _file:
        :return:
        """
        with open(path.join(_path, _file)) as src:
            rdr = reader(src)
            header = rdr.next()
            selector = [1] + range(4, len(header))
            # I need lanes 1+2 to build mappings of the locations and centromeres
            selector2 = [1, 2]
            header = np.array(header)[selector]
            locuses = []
            locus_locations = []
            for row in rdr:
                row = ['nan' if elt == 'NA' else elt for elt in row]
                locuses.append(np.genfromtxt(np.array(row))[selector].astype(np.float64))
                locus_locations.append(np.genfromtxt(np.array(row)[selector2]).astype(np.int32))


        # Recovers the chromosome limits
        locuses = np.array(locuses)
        locus_locations = np.array(locus_locations)

        chr_arr = locuses[:, 0]
        chr_brps = pull_breakpoints(chr_arr)
        broken_table = []  # broken tables has "true" set onto the array to the span of tag indexes indicating
                                # where a chromosome is present.
        chroms = range(int(np.min(chr_arr)), int(np.max(chr_arr))+1)
        for i in chroms:
            broken_table.append(chr_arr == i)                   # TODO: switch to the chr_brps
        broken_table = np.array(broken_table)
        chromosome_tag = np.repeat(locuses[:, 0].reshape((1, locuses.shape[0])), 200, axis=0)

        centromere_brps = position_centromere_breakpoints(get_centromeres(), locus_locations, broken_table)

        # segmentation code
        segments = np.concatenate((chr_brps, centromere_brps))
        segments = np.sort(segments).tolist()
        segments.insert(0, 0)
        segments.append(len(locuses)) # -1 ?
        segments = np.array([segments[:-1], segments[1:]]).T

        retdict = {
                'header': header,
                'chr_brps': chr_brps,
                'broken_table': broken_table,                   # TODO: switch to the chr_brps => Convert to bi-edges
                'chromosome_tag': chromosome_tag,
                'centromere_brps': centromere_brps,
                'chr_arr': chr_arr,
                'chroms': chroms,
                'locuses': locuses,
                'locus_locations': locus_locations,
                'segments': segments,
                    }
        return retdict


    def __str__(self):
        render_string = []
        for property, value in vars(self).iteritems():
            render_string.append( str(property)+": "+str(value))
        return "\n".join(render_string)



    def t_statistic_sorter(self, current_lane, breakpoints=None):
        """
        Collapses the segments of chromosomes whose levels are not statistically significantly different into a single level

        :param current_lane:
        :param breakpoints:
        :return:
        """
        t_mat = t_test_matrix(KS.brp_retriever(current_lane, breakpoints))

        t_mat[np.isnan(t_mat)] = 0
        t_mat = t_mat + t_mat.T
        np.fill_diagonal(t_mat, 1)
        ct_mat = t_mat.copy()
        ct_mat[t_mat < 0.01] = 0.01
        ct_mat = 1 - ct_mat

        # until here: prepares a matrix on which clustering is done

        Y = sch.linkage(ct_mat, method='centroid')
        clust_alloc = sch.fcluster(Y, 0.95, criterion='distance')
        # performs a hierarchical clustering

        averages = []
        # if HMM detected nothing, perform a swap
        if breakpoints is None:
            for i in range(0, 24): #separate treatments for chromosomes here
                averages.append(np.median(rm_nans(current_lane[self.broken_table[i]])))  # TODO: switch to the chr_brps
        else:
            subsets = np.split(current_lane, breakpoints[:-1])
            for subset in subsets:
                av = np.average(rm_nans(subset))
                averages.append(av)

        accumulator = [[] for _ in range(0, max(clust_alloc)+1)]
        for loc, item in enumerate(averages):
            accumulator[clust_alloc[loc]].append(item)

        # here, we just split the means that are sufficiently close statistically
        accumulator = np.array([ np.average(np.array(_list)) for _list in accumulator][1:])

        splitting_matrix = np.repeat(accumulator.reshape((1, accumulator.shape[0])),
                                        accumulator.shape[0], axis = 0)
        splitting_matrix = np.abs(splitting_matrix - splitting_matrix.T)

        show_2d_array(ct_mat, 'corrected p_values_matrix')
        show_2d_array(splitting_matrix, 'splitting matrix')

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
            re_chromosome_pad = np.repeat(self.chr_arr.reshape((1, self.chr_arr.shape[0])), 100, axis=0)
        else:
            pre_array = generate_breakpoint_mask(breakpoints)
            re_chromosome_pad = np.repeat(pre_array.reshape((1, pre_array.shape[0])), 100, axis=0)
            re_chromosome_pad += 1

        re_classification_tag = np.zeros(re_chromosome_pad.shape)

        for i in range(0, len(clust_alloc)):
            re_classification_tag[re_chromosome_pad == i+1] = sorter[clust_alloc_2[clust_alloc[i]-1]-1]

        if breakpoints:
            re_classification_tag = center_and_rebalance_tags(re_classification_tag)

        return re_classification_tag


    def compute_karyotype(self, current_lane, plotting=False):

        current_lane = current_lane - np.nanmean(current_lane) # sets the mean to 0

        # binariztion, plus variables for debugging.
        binarized, gauss_convolve, rolling_std, threshold = KS.binarize(current_lane)

        # actual location of HMM execution
        parsed = np.array(hmm.viterbi(parsing_hmm, initial_dist, binarized))

        # if we do a t_test_reduction, we need to do it here

        # we perform the injection of chr breakpoints here
        breakpoints = pull_breakpoints(parsed)
        # TODO: revert the breakpoint injection here.
        # it seems like a pretty bad idea, provided that the normalization brought this way might lead to loss of
        # segmental loss or gain detected by an hmm, the basis of our method.
        # PS: it also makes no sense methodologically

        # Nonetheless, main problems still remain in the reduction regression levels

        breakpoints = sorted(list(set(breakpoints + self.chr_brps + self.centromere_brps + [current_lane.shape[0]])))

        # we then compute per-breakpoints averages
        averages, stable_averages = KS.t_test_collapse(KS.brp_retriever(current_lane, breakpoints))
        segment_averages = KS.brp_setter(breakpoints, averages)
        stable_segment_averages = KS.brp_setter(breakpoints, stable_averages)

        # and we plot the classification debug plot if debug level is 2 or above
        if self.debug_level > 1:
            plot_classification(parsed, self.chromosome_tag, current_lane, gauss_convolve, rolling_std,
                                segment_averages, stable_segment_averages, threshold, plotting)


        # we compute the chromosome amplification decision here
        lw = [np.percentile(x, 25) for x in KS.brp_retriever(parsed, self.chr_brps)]
        hr = [np.percentile(x, 75) for x in KS.brp_retriever(parsed, self.chr_brps)]
        chromosome_state = [KS.support_function(lw_el, hr_el) for lw_el, hr_el in zip(lw, hr)]

        return current_lane - segment_averages, segment_averages, parsed, np.array(chromosome_state)


    def compute_recursive_karyotype(self, lane, plotting=False, debug_plotting=False):

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
                breakpoint_accumulator.append(breakpoint - prv_brp)
                prv_brp = breakpoint
            breakpoint_accumulator = np.array(breakpoint_accumulator)
            ############################################################
            msk = breakpoint_accumulator < self.min_seg_support
            shortness = inflate_support(current_lane.shape[0], breakpoints, msk)

            shortness_breakpoints = pull_breakpoints(shortness[0, :])

            chr_brp_arr = np.array(sorted(self.chr_brps + self.centromere_brps))
            re_brp = []
            supression_set = []
            prv_brp = 0

            # adds chromosome end breakpoints into breakpoint list
            for breakpoint in shortness_breakpoints:
                chr_break_verify = np.logical_and(chr_brp_arr > prv_brp, chr_brp_arr < breakpoint)
                if any(chr_break_verify) and all(shortness[0, :][prv_brp:breakpoint]):
                    re_brp += chr_brp_arr[chr_break_verify].tolist()
                    if re_brp[0] - prv_brp < self.collapse_segs:
                        supression_set.append(prv_brp)
                    if breakpoint - re_brp[-1] < self.collapse_segs:
                        supression_set.append(breakpoint)
                prv_brp = breakpoint
            shortness_breakpoints = sorted(shortness_breakpoints + re_brp)
            print 'supset', supression_set
            shortness_breakpoints = sorted(list(set(shortness_breakpoints) - set(supression_set)))

            shortness_breakpoints.append(shortness.shape[1])
            shortness_breakpoints = sorted(list(set(shortness_breakpoints)))
            # =>

            shortness_ladder = inflate_support(current_lane.shape[0], shortness_breakpoints)[0, :]

            filled_in = re_class_tag.copy().astype(np.float)
            levels = amplicons.copy()
            prv_brp = 0
            non_short_selector = np.logical_not(shortness[0, :].astype(np.bool))

            # ????
            for _i, breakpoint in enumerate(shortness_breakpoints):
                if all(shortness[0, :][prv_brp:breakpoint]):
                    current_fltr = shortness_ladder == _i
                    diff_min_max = np.max(parsed[current_fltr]) - np.min(parsed[current_fltr])
                    if diff_min_max == 0 and prv_brp - 1 > 0 and breakpoint + 1 < shortness_breakpoints[-1]:
                        non_modified = True
                        if parsed[prv_brp-1] == parsed[breakpoint+1]:
                            filled_in[:, current_fltr] = parsed[prv_brp - 1]
                            levels[current_fltr] = amplicons[prv_brp - 1]
                            non_modified = False
                        if prv_brp in self.chr_brps and breakpoint not in self.chr_brps: #TODO: add range+collapse
                            filled_in[:, current_fltr] = parsed[breakpoint + 1]
                            levels[current_fltr] = amplicons[breakpoint + 1]
                            non_modified = False
                        if breakpoint in self.chr_brps and prv_brp not in self.chr_brps: #TODO: add range+collapse
                            filled_in[:, current_fltr] = parsed[prv_brp - 1]
                            levels[current_fltr] = amplicons[prv_brp - 1]
                            non_modified = False
                        if non_modified:
                            cur_chr = self.chr_arr[breakpoint]
                            chr_median = np.median(amplicons[np.logical_and(cur_chr, non_short_selector)])
                            lar = np.array([prv_brp-1, breakpoint+1])
                            lval = amplicons[lar]
                            vl = np.argmin(lval-chr_median)
                            filled_in[:, current_fltr] = parsed[lar[vl]]
                            levels[current_fltr] = amplicons[lar[vl]]
                    else :
                        average = np.average(amplicons[current_fltr])
                        closest_index = np.argmin(np.abs(amplicons[non_short_selector] - average))
                        closest_index = np.array(range(0, shortness_breakpoints[-1]))[non_short_selector][closest_index]
                        color = parsed[closest_index]
                        filled_in[:, current_fltr] = color
                        levels[current_fltr] = amplicons[closest_index]
                prv_brp = breakpoint
            # =>

            filled_in = center_and_rebalance_tags(filled_in)
            levels = recompute_level(filled_in[0, :], levels)

            return  shortness, filled_in, levels


        current_lane = self.locuses[:, lane]
        retlist = self.compute_karyotype(current_lane, plotting=debug_plotting)

        amplicons = retlist[1]
        ampli_levels = retlist[2]-1
        re_retlist = copy.deepcopy(retlist)

        # this is where the computation of recursion is happening.
        for i in range(0, 6):
            re_retlist = self.compute_karyotype(re_retlist[0], plotting=debug_plotting)
            if np.max(re_retlist[2])-np.min(re_retlist[2]) < 1:
                break
            else:
                amplicons += re_retlist[1]
                ampli_levels += re_retlist[2] - 1

        breakpoints = pull_breakpoints(ampli_levels)
        # the next couple of lines stabilizes the nan-level removal
        brp_def_segs = np.split(np.array(current_lane), np.array(breakpoints))
        selection_mask = np.logical_not(np.array([np.isnan(np.nanmean(arr)) for arr in brp_def_segs]))
        breakpoints = np.array(breakpoints)[selection_mask[:-1]].tolist()

        breakpoints.append(current_lane.shape[0])

        re_class_tag = self.t_statistic_sorter(current_lane, breakpoints)
        parsed = re_class_tag[0, :]
        local, background, corrected_levels = determine_locality()

        parsed = background[0, :]
        collector = []
        for i in self.chroms:
            lw = np.percentile(parsed[self.broken_table[i-1, :]], 25)  # TODO: switch to the chr_brps
            hr = np.percentile(parsed[self.broken_table[i-1, :]], 75)  # TODO: switch to the chr_brps
            collector.append(support_function(lw, hr))

        collector2 = []
        for segment in self.segments:
            if segment[1]-segment[0] > 0:
                lw = np.percentile(parsed[segment[0]:segment[1]], 25)
                hr = np.percentile(parsed[segment[0]:segment[1]], 75)
                collector2.append(support_function(lw, hr))
            else:
                collector2.append(np.nan)

        if plotting:
            ax1 = plt.subplot(511)
            plt.imshow(self.chromosome_tag, interpolation='nearest', cmap='spectral')
            plt.imshow(background, interpolation='nearest', cmap='coolwarm')
            plt.setp(ax1.get_xticklabels(), fontsize=6)

            ax2 = plt.subplot(512, sharex=ax1)
            plt.plot(self.locuses[:, lane]-np.average(rm_nans(self.locuses[:, lane])), 'k.')
            plt.plot(amplicons, 'r', lw=2)
            plt.setp(ax2.get_xticklabels(), visible=False)

            ax3 = plt.subplot(513, sharex=ax1, sharey=ax2)
            plt.plot(corrected_levels, 'r', lw=2)
            plt.setp(ax3.get_xticklabels(), visible=False)

            ax4 = plt.subplot(514, sharex=ax1, sharey=ax2)
            plt.plot(amplicons-corrected_levels, 'g', lw=2)
            plt.setp(ax4.get_xticklabels(), visible=False)

            ax5 = plt.subplot(515, sharex=ax1)
            plt.setp(ax5.get_xticklabels(), visible=False)
            plt.imshow(self.chromosome_tag, interpolation='nearest', cmap='spectral')
            plt.imshow(inflate_support(self.chromosome_tag.shape[1], self.chr_brps, np.array(collector)),
                       interpolation='nearest', cmap='coolwarm', vmin=-1., vmax=1 )

            plt.show()

        return collector, background[0, :], collector2, amplicons-background[0, :]


    def compute_all_karyotypes(self):

        def plot(_list):
            plt.imshow(_list, interpolation='nearest', cmap='coolwarm')
            plt.show()

        def plot2(_list):
            inflated_table = np.vstack([inflate_tags(x[0, :], 25) for x in np.split(_list, _list.shape[0])])
            plt.imshow(inflated_table, interpolation='nearest', cmap='coolwarm')
            show_breakpoints(self.chr_brps, 'k')
            show_breakpoints(list(set(self.centromere_brps) - set(self.chr_brps)), 'g')
            plt.show()

        chromosome_list = []
        background_list = []
        arms_list = []
        all_breakpoints = []
        remainders_list = []
        for i in range(1, environment.locuses.shape[1]):
            print 'analyzing sample #', i
            col, bckg, col2, rmndrs = self.compute_recursive_karyotype(i, plotting=True)
            chromosome_list.append(col)
            background_list.append(bckg)
            arms_list.append(col2)
            all_breakpoints.append(pull_breakpoints(bckg))
            remainders_list.append(center_and_rebalance_tags(np.array(rmndrs).astype(np.float64)))
        chromosome_list = np.array(chromosome_list).astype(np.float64)
        background_list = np.vstack(tuple(background_list)).astype(np.float64)
        arms_list = np.array(arms_list).astype(np.float64)
        remainders_list = np.array(remainders_list).astype(np.float64)

        plot(chromosome_list)
        plot2(background_list)
        plot2(remainders_list)
        plot(arms_list)

        cell_line_dict = {}
        for background, arms, breakpoints, cell_line_name in zip(background_list.tolist(), arms_list.tolist(),
                                                                 all_breakpoints, self.header):
            cell_line_dict[cell_line_name] = {}
            breakpoints = np.array(breakpoints)
            background = np.array(background)
            for chromosome in range(0, self.broken_table.shape[0]):
                chr_p_dict = {}
                chr_q_dict = {}
                p_arm = self.segments[chromosome*2].tolist()
                q_arm = self.segments[chromosome*2 + 1].tolist()
                p_brps = breakpoints[np.logical_and(breakpoints > p_arm[0], breakpoints < p_arm[1])]
                q_brps = breakpoints[np.logical_and(breakpoints > q_arm[0], breakpoints < q_arm[1])]
                chr_p_dict['arm_level'] = arms[chromosome*2]
                chr_q_dict['arm_level'] = arms[chromosome*2 + 1]
                if p_arm[1] - p_arm[0] > 0 and p_brps.size:
                    p_characteristics = [[0] + self.locus_locations[p_brps].T[1].tolist(),
                    [background[p_arm[0] + 1].tolist()] + background[p_brps + 1].tolist()]
                else:
                    p_characteristics = [[], []]
                if q_arm[1] - q_arm[0] > 0 and q_brps.size:
                    q_characteristics = [[self.locus_locations[q_arm[0]][1]] + self.locus_locations[q_brps].T[1].tolist(),
                    [background[q_arm[0] + 1].tolist()] + background[q_brps + 1].tolist()]
                else:
                    q_characteristics = [[], []]
                chr_p_dict['segmental_aneuploidy'] = p_characteristics
                chr_q_dict['segmental_aneyploidy'] = q_characteristics
                cell_line_dict[cell_line_name][str(chromosome+1)+'-p'] = chr_p_dict
                cell_line_dict[cell_line_name][str(chromosome+1)+'-q'] = chr_q_dict

        return cell_line_dict


if __name__ == "__main__":
    pth = 'C:\\Users\\Andrei\\Desktop'
    fle = 'mmc2-karyotypes.csv'
    environment = Environement(pth, fle, debug_level=2)
    # print environment
    # print environment.compute_recursive_karyotype(40, True)
    pprint(environment.compute_all_karyotypes())