__author__ = 'Andrei'

import numpy as np
from matplotlib import pyplot as plt
import Karyotype_support as KS


def multilane_plot(main_pad, multi_pad_list):

    def morph_shape(arr, size):
        return np.repeat(arr[np.newaxis, :], size, axis=0)

    step_size = 200/(1+len(multi_pad_list))

    plt.imshow(morph_shape(main_pad, 200), interpolation='nearest', cmap='spectral')
    for i, array in enumerate(multi_pad_list):
        j = len(multi_pad_list) - i
        plt.imshow(morph_shape(array, j*step_size),interpolation='nearest', cmap='coolwarm', vmin=-1, vmax=1)


def remainder_plot(remainders):
    plt.plot(remainders, 'k.')
    lo, ho = KS.Tukey_outliers(remainders, FDR=0.0001)
    outliers = np.empty_like(remainders)
    outliers.fill(np.nan)
    outliers[ho] = remainders[ho]
    outliers[lo] = remainders[lo]
    plt.plot(outliers, 'r.')


def plot_classification(parsed, chr_tag, current_lane, segment_averages, binarized):

    ax1 = plt.subplot(311)
    multilane_plot(chr_tag, [parsed, binarized])
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax2 = plt.subplot(312, sharex=ax1)
    remainder_plot(current_lane)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.subplot(313, sharex=ax1)
    remainder_plot(current_lane - segment_averages)

    plt.show()


def multi_level_plot(chr_tag, starting_dataset, regression, final_remainder,
                     list_of_regressions, HMM_decisions, remainder_list,
                     HMM_states, chromosome_state, arms_state):

    ax1 = plt.subplot(511)
    remainder_plot(final_remainder)
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax2 = plt.subplot(512, sharex=ax1)
    plt.plot(starting_dataset, 'k.')
    plt.plot(regression)
    plt.setp(ax2.get_xticklabels(), fontsize=6)

    ax3 = plt.subplot(513, sharex=ax1)
    multilane_plot(chr_tag, list_of_regressions)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.ylim(0, 200)

    ax4 = plt.subplot(514, sharex=ax1)
    multilane_plot(chr_tag, HMM_decisions)
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.ylim(0, 200)

    ax5 = plt.subplot(515, sharex=ax1)
    multilane_plot(chr_tag, remainder_list)
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.ylim(0, 200)

    plt.show()

    ax1 = plt.subplot(311)
    plt.plot(starting_dataset, 'k.')
    plt.plot(regression)
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax2 = plt.subplot(312, sharex=ax1)
    multilane_plot(chr_tag, [HMM_states])
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylim(0, 200)

    ax3 = plt.subplot(313, sharex=ax1)
    multilane_plot(chr_tag, [chromosome_state, arms_state])
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.ylim(0, 200)

    plt.show()


def old_multi_level_plot():
    ax1 = plt.subplot(511)
    plt.imshow(self.chromosome_tag, interpolation='nearest', cmap='spectral')
    plt.imshow(background, interpolation='nearest', cmap='coolwarm')
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax2 = plt.subplot(512, sharex=ax1)
    plt.plot(self.locuses[:, lane]-np.nanmean(self.locuses[:, lane]), 'k.')
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


def plot(_list):
    plt.imshow(_list, interpolation='nearest', cmap='coolwarm')
    plt.show()


def plot2(_list):
    inflated_table = np.vstack([inflate_tags(x[0, :], 25) for x in np.split(_list, _list.shape[0])])
    plt.imshow(inflated_table, interpolation='nearest', cmap='coolwarm')
    show_breakpoints(self.chr_brps, 'k')
    show_breakpoints(list(set(self.centromere_brps) - set(self.chr_brps)), 'g')
    plt.show()