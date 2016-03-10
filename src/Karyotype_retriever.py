from csv import reader
from os import path
import numpy as np
from pprint import pprint
from chiffatools import hmm
from src.pre_processors import get_centromeres
from src.supporting_functions import pull_breakpoints, center_and_rebalance_tags, position_centromere_breakpoints, HMM_constructor
import src.supporting_functions as sf
from src.drawing_functions import plot_classification, multi_level_plot, plot, plot2
from src.basic_drawing import show_2d_array


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
    Just a payload object to carry around environment variables that are specific
     to the instance that is getting executed
    """

    def __init__(self, _path, _file, min_seg_support=25,
                 collapse_segs=3, debug_level=0,
                 coarse_hmm_parms=(10, 6, 0.6),
                 fine_hmm_params=(3, 3, 0.1)):

        self.header = None
        self.chr_breakpoints = None
        self.broken_table = None
        self.chromosome_tag = None
        self.centromere_breakpoints = None
        self.locus_locations = None
        self.chr_arr = None
        self.chromosomes = None
        self.locuses = None
        self.segments = None
        self.strains = None
        self.min_seg_support = min_seg_support
        self.collapse_segments = collapse_segs
        self.debug_level = debug_level
        self.coarse_pass_parameters = coarse_hmm_parms
        self.fine_pass_parameters = fine_hmm_params

        if _file is not None:
            parameter_dict = self.wrap_import(_path, _file)
            for name, value in parameter_dict.iteritems():
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
            print 'header: %s' % header
            self.strains = header[4:]
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
        self.locus_locations = locus_locations

        chr_arr = locuses[:, 0]
        chr_brps = pull_breakpoints(chr_arr)
        broken_table = []
        # broken tables has "true" set onto the array to the span of tags where a chromosome is
        chromosomes = range(int(np.min(chr_arr)), int(np.max(chr_arr))+1)
        for i in chromosomes:
            broken_table.append(chr_arr == i)                 # TODO: switch to the chr_breakpoints
        broken_table = np.array(broken_table)
        chromosome_tag = np.repeat(locuses[:, 0].reshape((1, locuses.shape[0])), 200, axis=0)

        centromere_brps = position_centromere_breakpoints(get_centromeres(),
                                                          locus_locations, broken_table)

        # segmentation code
        segments = np.concatenate((chr_brps, centromere_brps))
        segments = np.sort(segments).tolist()
        segments.insert(0, 0)
        segments.append(len(locuses)) # -1 ?
        segments = np.array([segments[:-1], segments[1:]]).T

        retdict = {
                'header': header,
                'chr_breakpoints': chr_brps,
                'broken_table': broken_table,
                # TODO: switch to the chr_breakpoints => Convert to bi-edges
                'chromosome_tag': chromosome_tag,
                'centromere_breakpoints': centromere_brps,
                'chr_arr': chr_arr,
                'chromosomes': chromosomes,
                'locuses': locuses,
                'locus_locations': locus_locations,
                'segments': segments, }
        return retdict

    def __str__(self):
        render_string = []
        for _property, _value in vars(self).iteritems():
            render_string.append(str(_property)+": "+str(_value))
        return "\n".join(render_string)

    def hmm_regress(self, current_lane, coherence_length=10, fdr=0.01):
        """
        Performs a regression of current lane by an hmm with a defined coherence length

        :param current_lane:
        :param coherence_length:
        :param fdr:
        :return:
        """
        parsing_hmm = HMM_constructor(coherence_length)

        # normalization of the mean
        current_lane = current_lane - np.nanmean(current_lane)

        # binariztion, plus variables for debugging.
        binarized = sf.binarize(current_lane, fdr)
        # sf.probabilistic_binarize(current_lane, fdr)

        # actual location of HMM execution
        parsed = np.array(hmm.viterbi(parsing_hmm, initial_dist, binarized)) - 1

        # segment_averages = sf.old_padded_means(current_lane, parsed)
        segment_averages = sf.padded_means(current_lane, parsed)

        # and we plot the classification debug plot if debug level is 2 or above
        if self.debug_level > 1:
            plot_classification(parsed, self.chromosome_tag[0, :], current_lane,
                                segment_averages, binarized - 1, fdr)

        # we compute the chromosome amplification decision here. TODO: move for recursive conclusion
        # lw = [np.percentile(x, 25) for x in sf.brp_retriever(parsed, self.chr_breakpoints)]
        # hr = [np.percentile(x, 75) for x in sf.brp_retriever(parsed, self.chr_breakpoints)]
        # chromosome_state = [sf.support_function(lw_el, hr_el) for lw_el, hr_el in zip(lw, hr)]

        # print sf.model_stats(current_lane, segment_averages)
        return current_lane - segment_averages, segment_averages, parsed

    def recursive_hmm_regression(self, lane):

        def regression_round(coherence_length, max_rounds, fdr):

            for i in range(0, max_rounds):
                reg_remainder, hmm_reg, hmm_levels = self.hmm_regress(reg_remainders[-1],
                                                                      coherence_length, fdr)
                reg_stats = sf.model_stats(reg_remainders[-1], hmm_reg)
                # print reg_stats
                if not sf.model_decision(*reg_stats):
                    # print 'break'
                    break
                else:
                    reg_remainders.append(reg_remainder)
                    hmm_regressions.append(hmm_reg)
                    hmm_level_decisions.append(hmm_levels)

        current_lane = self.locuses[:, lane]

        reg_remainders = [current_lane]
        hmm_regressions = []
        hmm_level_decisions = []

        # this is where the computation of recursion is happening.
        regression_round(*self.coarse_pass_parameters)

        # make a final fine-grained parse
        regression_round(*self.fine_pass_parameters)

        if hmm_level_decisions == []:
            hmm_regressions = [np.zeros_like(current_lane)]
            hmm_level_decisions = [np.zeros_like(current_lane)]

        final_regression = np.array(hmm_regressions).sum(0)
        # final_regression = center_and_rebalance_tags(final_regression)
        #  TODO: technically, this time we can use the fact that HMM = 0 means base state

        final_HMM = np.array(hmm_level_decisions).sum(0).round()
        final_HMM[final_HMM > 1] = 1
        final_HMM[final_HMM < -1] = -1
        # TODO: correct computation of the HMM decision: current one is pretty much a fail

        lw = [np.percentile(x, 25) for x in sf.brp_retriever(final_HMM, self.chr_breakpoints)]
        hr = [np.percentile(x, 75) for x in sf.brp_retriever(final_HMM, self.chr_breakpoints)]
        chromosome_state = [sf.support_function(lw_el, hr_el) for lw_el, hr_el in zip(lw, hr)]
        chr_state_pad = sf.brp_setter(self.chr_breakpoints + [current_lane.shape[0]], chromosome_state)

        lw = [np.percentile(x, 25) for x in sf.brp_retriever(final_HMM, self.chr_breakpoints +
                                                             self.centromere_breakpoints)]
        hr = [np.percentile(x, 75) for x in sf.brp_retriever(final_HMM, self.chr_breakpoints +
                                                             self.centromere_breakpoints)]
        arms_state = [sf.support_function(lw_el, hr_el) for lw_el, hr_el in zip(lw, hr)]
        arms_state_pad = sf.brp_setter(self.chr_breakpoints + self.centromere_breakpoints +
                                       [current_lane.shape[0]], arms_state)

        if self.debug_level > 0:
            multi_level_plot(self.chromosome_tag[0, :], current_lane,
                             final_regression, reg_remainders[-1],
                             hmm_regressions, hmm_level_decisions, reg_remainders,
                             final_HMM, chr_state_pad, arms_state_pad)

        outliers = sf.get_outliers(reg_remainders[-1], 0.005)
        outliers[np.isnan(outliers)] = 0

        return chromosome_state, final_regression, arms_state, outliers

    def compute_all_karyotypes(self):

        chromosome_list = []
        background_list = []
        arms_list = []
        all_breakpoints = []
        remainders_list = []

        for i in range(1, self.locuses.shape[1]):
            print 'Environement.compute_all_karyotypes analyzing ', self.strains[i-1]
            col, bckg, col2, rmndrs = self.recursive_hmm_regression(i)
            chromosome_list.append(col)
            background_list.append(sf.center_and_rebalance_tags(bckg))
            arms_list.append(col2)
            all_breakpoints.append(pull_breakpoints(bckg))
            remainders_list.append(sf.center_and_rebalance_tags(rmndrs))

        chromosome_list = np.array(chromosome_list).astype(np.float64)
        background_list = np.vstack(tuple(background_list)).astype(np.float64)
        arms_list = np.array(arms_list).astype(np.float64)
        remainders_list = np.array(remainders_list).astype(np.float64)

        # plot(chromosome_list)
        plot2(background_list, self.chr_breakpoints, self.centromere_breakpoints, self.strains)
        plot2(remainders_list, self.chr_breakpoints, self.centromere_breakpoints, self.strains)
        # plot(arms_list)

        chr_arm_locations, chr_arm_names = sf.align_chromosome_edges(self.chr_breakpoints,
                                                                     self.centromere_breakpoints)

        # export of the data starts from here
        cell_line_dict = {}
        cell_line_dict["meta"] = {
            "strains": self.strains,
            "chromosome arms": (chr_arm_locations, chr_arm_names),
            "HMM CNV matrix": background_list,
            "HMM remainders CNV matrix": remainders_list,
            "locus locations": self.locus_locations,
        }

        # for background, arms, breakpoints, cell_line_name in zip(background_list.tolist(),
        #                                                          arms_list.tolist(),
        #                                                          all_breakpoints, self.header):
        #     cell_line_dict[cell_line_name] = {}
        #     breakpoints = np.array(breakpoints)
        #     background = np.array(background)
        #     for chromosome in range(0, self.broken_table.shape[0]):
        #         chr_p_dict = {}
        #         chr_q_dict = {}
        #         p_arm = self.segments[chromosome*2].tolist()
        #         q_arm = self.segments[chromosome*2 + 1].tolist()
        #         p_brps = breakpoints[np.logical_and(breakpoints > p_arm[0], breakpoints < p_arm[1])]
        #         q_brps = breakpoints[np.logical_and(breakpoints > q_arm[0], breakpoints < q_arm[1])]
        #         chr_p_dict['arm_level'] = arms[chromosome*2]
        #         chr_q_dict['arm_level'] = arms[chromosome*2 + 1]
        #         if p_arm[1] - p_arm[0] > 0 and p_brps.size:
        #             p_characteristics = [[0] + self.locus_locations[p_brps].T[1].tolist(),
        #                 [background[p_arm[0] + 1].tolist()] + background[p_brps + 1].tolist()]
        #         else:
        #             p_characteristics = [[], []]
        #         if q_arm[1] - q_arm[0] > 0 and q_brps.size:
        #             q_characteristics = [[self.locus_locations[q_arm[0]][1]] +
        #                                  self.locus_locations[q_brps].T[1].tolist(),
        #                 [background[q_arm[0] + 1].tolist()] + background[q_brps + 1].tolist()]
        #         else:
        #             q_characteristics = [[], []]
        #         chr_p_dict['segmental_aneuploidy'] = p_characteristics
        #         chr_q_dict['segmental_aneyploidy'] = q_characteristics
        #         cell_line_dict[cell_line_name][str(chromosome+1)+'-p'] = chr_p_dict
        #         cell_line_dict[cell_line_name][str(chromosome+1)+'-q'] = chr_q_dict

        return cell_line_dict

if __name__ == "__main__":
    pth = 'C:\\Users\\Andrei\\Desktop'
    fle = 'mmc2-karyotypes.csv'
    # fle2 = 'mmc1-karyotypes.csv'
    # TODO: mechanism for 22 chromosomes instead of 24
    environment = Environement(pth, fle, debug_level=1,
                               coarse_hmm_parms=(10, 6, 0.6), fine_hmm_params=(3, 3, 0.1))
    # print environment
    # print environment.recursive_hmm_regression(42)
    print environment.recursive_hmm_regression(43)
    # print environment.recursive_hmm_regression(44)
    # print environment.compute_all_karyotypes()
    # currently violating: # 38

    # probabilistic binarization => avoid slight off-set between lanes just on the edge of first regression
    # AIC that does not take any stopping criteria in account creating a perfect regression