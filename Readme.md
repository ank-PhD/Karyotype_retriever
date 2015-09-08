[![License Type](https://img.shields.io/badge/license-BSD3-blue.svg)][https://github.com/chiffa/Karyotype_retriever/blob/master/License-BSD3]

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
 * click
 * chiffatools


chiffatools can be downloaded and installed via
    > pip install git+https://github.com/ciffa/chiffatools

Pay attention, you will need to manually select the file you are willing to analyze and type it's name in the
source code for now.

In order to execute the pipeline, setup the environement with a
```
    > import Karyotype_rertiever as KR
    > env = KR.Environement(path_to_your_file, file_name)
    > result = env.compute_all_karyotypes()
```

In order, you will be shown gain/losses on the chromosome level, label level and chromosome level.

Alternatively, if the script was installed with pip:
```
    > Karyotype_retriever run_pipeline path_to_your_file file_name
```

The result is a 3-level dict:
```
    { <Strain Name>:
        {<chromosome#>-p/q:
            {'arm_level': arm-level aneuploidy,
             'segmental_aneuploidy': [[positions of arm start and segmental aneuploidy breakpoints (kb)],
                                      [ploidy level after the breakpoint above]]
             }
        ...},
    ....}
```

The time of execution on a modern workstation, for a single sample of ~3000 genomic locations is less than a second

Chromosomes X and Y were re-named to 23 and 24 respectively.

**Algorithm description:**


**Static data files:**
 - CytoBands.txt courtesy of http://hgdownload.cse.ucsc.edu/goldenPath/hg18/database/cytoBand.txt.gz

**Future developments:**
 - Collapse HMM predictions onto a chromosome limits or centromere limits if the transition boundaries are close (critcial)
 - Factor out the cenromere collapse paramters and minimal width to be accessible by the user
 - Output the map of remainder amplifications 
 - Made the HMM aware of distances between locuses measured on affymetrix chip (1) and recombination hotspots (2)
 - Reformulate as Bayesian choice: state of markers =  evidence; distance = prob. of transition or collapse (?)
 - Implement clustering of cell lines on the level of chromosome gain/loss similarity (?)
