Python implementation of HMM regression of segmental ploidy state for DNA chip
===============================================================================================================

This 

This is a small tool based on the HMMs and t_test series to determine the most likely chromosome-level
karyotype of a sample while keeping the information about local amplifications.

As an input it requires a file that contains per row the following informations:
 - site ID
 - chromosome on which the site is
 - absolute genome location (required for ordering)
 - for each sample, relative gain/loss and 'nan' for the absent data.

First row is expected to be a set of sample identifiers, so that this information can be used in the subsequent
computations

Required packages:
 * numpy
 * scipy
 * matplotlib
 * chiffatools


chiffatools can be downloaded and installed via
    > pip install git+https://github.com/ciffa/chiffatools

Pay attention, you will need to manually select the file you are willing to analyze and type it's name in the
source code for now.

**Future developments:**
 - Made the HMM aware of distances between locuses measured on affymetrix plateform and transition between the chromosomes