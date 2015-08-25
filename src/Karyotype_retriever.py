__author__ = 'ank'

from csv import reader
from os import path
import numpy as np
from pprint import pprint
from chiffatools import hmm
from src.pre_processors import get_centromeres
from src.Karyotype_support import pull_breakpoints, center_and_rebalance_tags, position_centromere_breakpoints, HMM_constructor
import src.Karyotype_support as KS
from src.Drawing_functions import plot_classification, multi_level_plot, plot, plot2
from  src.basic_drawing import show_2d_array

# TODO: center and rebalance
# TODO: last HMM pass to group together the remainder outliers and the HMM tags that are separated by too short
# TODO: we still have gained/lost problem and overregression when FDR is set to 5%
# TODO: big problem: Tuckey has a solid cut-off during the binarization. Which means that small oscillations are really
#       dependnet on small level variations
# TODO: legacy centering procedure does not seem to work properly for some reason.


##################################################################################
# suppresses the noise from Numpy about future suppression of the boolean mask use
##################################################################################
# warnings.catch_warnings()
# warnings.simplefilter("ignore")

####################################################
# Defines the parsing HMM and matrices underlying it
####################################################
initial_dist = np.array([[0.33, 0.34, 0.33]])


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
        broken_table = []  # broken tables has "true" set onto the array to the span of tags where a chromosome is
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
                'segments': segments, }
        return retdict


    def __str__(self):
        render_string = []
        for _property, _value in vars(self).iteritems():
            render_string.append(str(_property)+": "+str(_value))
        return "\n".join(render_string)


    def hmm_regress(self, current_lane, coherence_length=10, FDR=0.01):
        """
        Performs a regression of current lane by an hmm with a defined coherence length

        :param current_lane:
        :param coherence_length:
        :return:
        """
        parsing_hmm = HMM_constructor(coherence_length)

        current_lane = current_lane - np.nanmean(current_lane)  # sets the mean to 0

        # binariztion, plus variables for debugging.
        binarized = KS.binarize(current_lane, FDR)

        # actual location of HMM execution
        parsed = np.array(hmm.viterbi(parsing_hmm, initial_dist, binarized)) - 1

        # segment_averages = KS.old_padded_means(current_lane, parsed)
        segment_averages = KS.padded_means(current_lane, parsed)

        # and we plot the classification debug plot if debug level is 2 or above
        if self.debug_level > 1:
            plot_classification(parsed, self.chromosome_tag[0, :], current_lane,
                                segment_averages, binarized-1, FDR)

        # we compute the chromosome amplification decision here. TODO: move for recursive conclusion
        # lw = [np.percentile(x, 25) for x in KS.brp_retriever(parsed, self.chr_brps)]
        # hr = [np.percentile(x, 75) for x in KS.brp_retriever(parsed, self.chr_brps)]
        # chromosome_state = [KS.support_function(lw_el, hr_el) for lw_el, hr_el in zip(lw, hr)]

        # print KS.model_stats(current_lane, segment_averages)
        return current_lane - segment_averages, segment_averages, parsed


    def recursive_hmm_regression(self, lane):

        def regression_round(coherence_lenght, max_rounds, FDR):
            for i in range(0, max_rounds):
                reg_remainder, HMM_reg, HMM_levels = self.hmm_regress(reg_remainders[-1], coherence_lenght, FDR)
                reg_stats = KS.model_stats(reg_remainders[-1], HMM_reg)
                print reg_stats
                if not KS.model_decision(*reg_stats):
                    print 'break'
                    break
                else:
                    reg_remainders.append(reg_remainder)
                    HMM_regressions.append(HMM_reg)
                    HMM_level_decisions.append(HMM_levels)


        current_lane = self.locuses[:, lane]

        reg_remainders = [current_lane]
        HMM_regressions = []
        HMM_level_decisions = []

        # this is where the computation of recursion is happening.
        regression_round(10, 6, 0.5)

        # make a final fine-grained parse
        regression_round(3, 3, 0.05)

        if HMM_level_decisions == []:
            HMM_regressions = [np.zeros_like(current_lane)]
            HMM_level_decisions = [np.zeros_like(current_lane)]

        final_regression = np.array(HMM_regressions).sum(0)
        # final_regression = center_and_rebalance_tags(final_regression) # TODO: technically, this time we can use the fact that HMM = 0 means base state

        final_HMM = np.array(HMM_level_decisions).sum(0).round()
        final_HMM[final_HMM > 1] = 1
        final_HMM[final_HMM < -1] = -1
        # TODO: correct computation of the HMM decision: current one is pretty much a fail

        lw = [np.percentile(x, 25) for x in KS.brp_retriever(final_HMM, self.chr_brps)]
        hr = [np.percentile(x, 75) for x in KS.brp_retriever(final_HMM, self.chr_brps)]
        chromosome_state = [KS.support_function(lw_el, hr_el) for lw_el, hr_el in zip(lw, hr)]
        chr_state_pad = KS.brp_setter(self.chr_brps + [current_lane.shape[0]], chromosome_state)

        lw = [np.percentile(x, 25) for x in KS.brp_retriever(final_HMM, self.chr_brps + self.centromere_brps)]
        hr = [np.percentile(x, 75) for x in KS.brp_retriever(final_HMM, self.chr_brps + self.centromere_brps)]
        arms_state = [KS.support_function(lw_el, hr_el) for lw_el, hr_el in zip(lw, hr)]
        arms_state_pad = KS.brp_setter(self.chr_brps + self.centromere_brps + [current_lane.shape[0]], arms_state)

        if self.debug_level > 0:
            multi_level_plot(self.chromosome_tag[0, :], current_lane, final_regression, reg_remainders[-1],
                             HMM_regressions, HMM_level_decisions, reg_remainders,
                             final_HMM, chr_state_pad, arms_state_pad)

        return chromosome_state, final_regression, arms_state, reg_remainders[-1]


    def compute_all_karyotypes(self):

        chromosome_list = []
        background_list = []
        arms_list = []
        all_breakpoints = []
        remainders_list = []
        for i in range(1, environment.locuses.shape[1]):
            print 'analyzing sample #', i
            col, bckg, col2, rmndrs = self.recursive_hmm_regression(i)
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
        plot2(background_list, self.chr_brps, self.centromere_brps)
        plot2(remainders_list, self.chr_brps, self.centromere_brps)
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
    print environment.recursive_hmm_regression(13)

    # TODO: attempt sliding window distribution?
    #       the idea is now pretty much to find a proper binarization technique

    # One of the investigated ways is to perform rolling means normalization
    # exclude all the outliers
    # perform the calculation of the rolling mean and dispersion
    # for rolls of contigs, peroform

    # Tuckey procedure works averagely because some chromosomes have really
    # low basis dispersion level, whereas the other have a higher, whereas
    # Tuckey assimilates them all into one.

    # Also a big problem is the solidity of threshold, creating discontinuities
    # just on the verge of threshold and separating statistically indistinguishable levels

    # => we could correct it by adding an instable intermediate level (can only last 2-3 datapoints)
    # to account for the ramp due to chromosome points breakage.

    # or entirely different: we can create a duplex HMM of "level change" and "is an outlier"
    # We use the quitilized probability that can get classified either as an outlier or as a
    # probability we are at a specific level. When a level chage occurs, we back-propagate to most likely
    # supporting switch.
    # we actually have now 2 2-level HMM that get updated due to pretty specific rules
    # if we use the estimate of mean of the current level (survival function with already known mean and
    # STD, we can get Proba of transition to a new model pretty easily)

    # in the end, we have a 2-state HMM with a more complex update rule, that gets reset to 0 every time
    # change is probably detected

    # The problem is that afterwards we loose the ability to detect the assumption that

    # we can though add an outlier HMM state, which is quite likely to be jumped into, but is unstable


    # to investigate: 5, 9, 10, 11, 12 => we need to perform a sliding normalization
    # pprint(environment.compute_all_karyotypes())