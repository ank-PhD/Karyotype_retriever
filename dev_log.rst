Developper's log:
=================

Confirmed implementation:
-------------------------

 - use the new arm map in order to properly cut the chromosome arms for the strains output.


Features:
---------

 - Collapse HMM predictions onto a chromosome limits or centromere limits if the transition
   boundaries are close (critcial)

 - Factor out the cenromere collapse paramters and minimal width to be accessible by the user

 - Output the map of remainder amplifications

 - Made the HMM aware of distances between locuses measured on affymetrix chip (1)
   and recombination hotspots (2)

 - Reformulate as Bayesian choice: state of markers =  evidence; distance = prob.
   of transition or collapse (?)

 - Implement clustering of cell lines on the level of chromosome gain/loss similarity (?)


Refactoring:
------------

 - unifromize the way NaNs are treated.