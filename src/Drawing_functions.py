__author__ = 'Andrei'

import numpy as np
from matplotlib import pyplot as plt

def show_2d_array(data, title=False):
    if title:
        plt.title(title)
    plt.imshow(data, interpolation='nearest', cmap='coolwarm')
    plt.colorbar()
    plt.show()


def plot_classification(parsed, chr_tag, current_lane, gauss_convolve, rolling_std, segment_averages, threshold, render):

    classification_tag = np.repeat(parsed.reshape((1, parsed.shape[0])), 100, axis=0)

    ax1 = plt.subplot(311)
    plt.imshow(chr_tag, interpolation='nearest', cmap='spectral')
    plt.imshow(classification_tag, interpolation='nearest', cmap='coolwarm', vmin=0, vmax=2)
    plt.setp(ax1.get_xticklabels(), fontsize=6)

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(current_lane, 'k.')
    plt.plot(gauss_convolve, 'r', lw=2)
    plt.plot(gauss_convolve + rolling_std, 'g', lw=1)
    plt.plot(gauss_convolve - rolling_std, 'g', lw=1)
    plt.plot(segment_averages, 'b', lw=2)
    plt.axhline(y=threshold, color='c')
    plt.axhline(y=-threshold, color='c')
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.subplot(313, sharex=ax1)
    plt.plot(current_lane - segment_averages, 'k.')

    plt.show()


def multi_level_plot():
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