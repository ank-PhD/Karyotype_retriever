import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import supporting_functions as KS
from chiffatools.dataviz import smooth_histogram
from scipy.stats import beta


def multilane_plot(main_pad, multi_pad_list):

    def morph_shape(arr, size):
        return np.repeat(arr[np.newaxis, :], size, axis=0)

    step_size = 200/(1+len(multi_pad_list))

    plt.imshow(morph_shape(main_pad, 200), interpolation='nearest', cmap='spectral')
    for i, array in enumerate(multi_pad_list):
        j = len(multi_pad_list) - i
        plt.imshow(morph_shape(array, j*step_size),interpolation='nearest', cmap='coolwarm', vmin=-1, vmax=1)


def remainder_plot(remainders, FDR=0.005):
    plt.plot(remainders, 'k.')
    plt.plot(KS.get_outliers(remainders, FDR), 'r.')


def plot_classification(parsed, chr_tag, current_lane, segment_averages, binarized, FDR):

    ax1 = plt.subplot(311)
    multilane_plot(chr_tag, [parsed, binarized])
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax2 = plt.subplot(312, sharex=ax1)
    remainder_plot(current_lane, FDR)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.subplot(313, sharex=ax1, sharey=ax2)
    remainder_plot(current_lane - segment_averages, FDR)

    plt.show()


def multi_level_plot(chr_tag, starting_dataset, regression, final_remainder,
                     list_of_regressions, HMM_decisions, remainder_list,
                     HMM_states, chromosome_state, arms_state,):

    ax1 = plt.subplot(511)
    remainder_plot(final_remainder, FDR=0.01)
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

    ax1 = plt.subplot(511)
    plt.plot(starting_dataset, 'k.')
    plt.plot(regression)
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax2 = plt.subplot(512, sharex=ax1)
    multilane_plot(chr_tag, [HMM_states])
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylim(0, 200)

    ax3 = plt.subplot(513, sharex=ax1)
    multilane_plot(chr_tag, [chromosome_state, arms_state])
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.ylim(0, 200)

    ax4 = plt.subplot(514, sharex=ax1)
    c_remainder = KS.get_outliers(final_remainder,0.005)
    c_remainder[np.isnan(c_remainder)] = 0
    multilane_plot(chr_tag, [regression, c_remainder])
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.ylim(0, 200)

    plt.show()


def show_breakpoints(breakpoints, color = 'k'):
    """
    plots the breakpoints

    :param breakpoints:
    :return:
    """
    for point in breakpoints:
        plt.axvline(x=point, color=color)


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


def plot(_list):
    plt.imshow(_list, interpolation='nearest', cmap='coolwarm')
    plt.show()


def plot2(_list, chr_brps, centromere_brps):
    inflated_table = np.vstack([inflate_tags(x[0, :], 25) for x in np.split(_list, _list.shape[0])])

    gs = gridspec.GridSpec(4, 4)

    ax1 = plt.subplot(gs[:-1, :])
    plt.imshow(inflated_table, interpolation='nearest', cmap='coolwarm')
    show_breakpoints(chr_brps, 'k')
    show_breakpoints(list(set(centromere_brps) - set(chr_brps)), 'g')

    ax2 = plt.subplot(gs[-1, :], sharex=ax1)
    red_run = np.nanmean((_list > 0).astype(np.float), 0)
    blue_run = np.nanmean((_list < 0).astype(np.float), 0)

    stack = np.hstack((blue_run, red_run))
    mean = np.mean(stack)
    std = np.std(stack)
    _alpha = ((1 - mean)/std**2 - 1/mean)*mean**2
    _beta = _alpha*(1/mean-1)
    r = beta.rvs(_alpha, _beta, size=1000)
    _min, _max = beta.interval(0.95, _alpha, _beta)

    plt.plot(blue_run, 'b')
    plt.plot(red_run, 'r')
    plt.axhline(y=_min, color='g')
    plt.axhline(y=_max, color='g')
    show_breakpoints(chr_brps, 'k')
    show_breakpoints(list(set(centromere_brps) - set(chr_brps)), 'g')
    plt.show()

    smooth_histogram(r, 'b')
    smooth_histogram(stack)
    plt.axvline(x=_max, color='g')
    plt.axvline(x=_min, color='g')
    plt.show()