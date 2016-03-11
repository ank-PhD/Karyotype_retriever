[![License Type](https://img.shields.io/badge/license-BSD3-blue.svg)](https://github.com/chiffa/Karyotype_retriever/blob/master/License-BSD3)

Python implementation of HMM regression of segmental ploidy state for DNA chip
===============================================================================

This is a small tool based on iterative HMM regression based on Tuckey quitile-based outlier 
detection. It allows to reconstruct the most likely segmental aneuploidy pattern for a sample 
while keeping the information about local aplifications.

**Source files:**

As an input it requires a file that contains per row the following information:
 - site ID
 - chromosome on which the site is
 - absolute genome location (required for ordering)
 - for each sample, relative gain/loss and 'nan' for the absent data.

First row is expected to be a set of sample identifiers, so that this information can be used in the subsequent
computations

**Installation:**

 * numpy
 * scipy
 * matplotlib
 * click
 * chiffatools

All packages but the `chiffatools` can be installed with pip:
```
    > pip install git+https://github.com/chiffa/Karyotype_retriever.git
```

`chiffatools` is a personal helper function library and can be installed with pip:
```
    > pip install git+https://github.com/ciffa/chiffatools
```

You will need to manually select the file you are willing to analyze and provide it's name upon 
calling the library or the shell interface.

**Usage**

In order to execute the pipeline, setup the environment and with the following commands:
```
    > import Karyotype_rertiever as KR
    > env = KR.Environement(path_to_your_file, file_name)
    > result = env.compute_all_karyotypes()
```

Additional parameters for `Environement` are available:
 - `debug_level` - can be set on 1 or 2, depending on the details of debug desired 
 - `coarse_hmm_parameters` - minimal elements triggering hmm switch, maximum iteration, FDR.
    Defaults to `(10, 6, 0.6)`.
 - `fine_hmm_parameters` -minimal elements triggering hmm switch, maximum iteration, FDR.  
    Defaults to `(3, 3, 0.1)`


The `result` is a dict containing a single "meta entry":
```
    {'HMM CNV matrix': numpy.array(columns=locuses, rows=strains),
     'HMM remainders CNV matrix': numpy.array(columns=locuses, rows=strains),
     'chromosome arms': ([locus# at chr arm breakpoints, ...],
                         [Chr arm names, ...]),
     'locus locations': numpy.array([[chr#,  location on chr (in kb)],...]),
     'strains': [strains list, ...]}
```

Alternatively, if the script was installed with pip command line interface is available from shell:
```
    > Karyotype_retriever run_pipeline path_to_your_file file_name
```

You can see an example of rendering shown before returning result below:

![Final output](http://i.imgur.com/URgjyRl.png)


If the argument `debug_level` is set to 1 or above in the `Environment` initialization, for each 
strain analyzed the following summary images will be shown. 

![Intermediate1](http://i.imgur.com/wflUrZg.png)

 - remainder plot after the regression
 - initial plot with the HMM regression
 - chromosomes with levels regressed upon each iteration
 - remainder of the regression after each iteration


![Intermediate2](http://i.imgur.com/7r9YPzG.png)

 - initial plot with HMM regression
 - gain/loss classifier
 - chromosome and chromosome arm gain/loss classifier
 - actual levels and remainders of segmental aneuploidy retrieval

The time of execution on a modern workstation, for a single sample of ~3000 genomic locations is less than a second

Chromosomes X and Y were re-named to 23 and 24 respectively.

**Algorithm description:**

We basically perform an HHM classifier after having thresholded the DNA abundance at each locus 
into three classes (low, normal, high) with a pre-defined FDR over all the data based on the 
first and third quantiles based on Tuckey outlier detection procedures

**Static data files:**

 - CytoBands.txt courtesy of [UCSC](http://hgdownload.cse.ucsc.edu/goldenPath/hg18/database/cytoBand.txt.gz)

**Future developments:**

 - Collapse HMM predictions onto a chromosome limits or centromere limits if the transition 
 boundaries are close (critical)
 - Factor out the centromere collapse parameters and minimal width to be accessible by the user
 - Output the map of remainder amplifications 
 - Made the HMM aware of distances between locuses measured on affymetrix chip (1) and recombination hotspots (2)
 - Reformulate as Bayesian choice: state of markers =  evidence; distance = prob. of transition or collapse (?)
 - Implement clustering of cell lines on the level of chromosome gain/loss similarity (?)
