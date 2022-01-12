# Reichenbach_et_al_2022
Python scripts used for Reichenbach et al 2022 (DOI: 10.7554/eLife.72668)

You can find here five Python scripts we used to analyse photometry data recorded with the RZ5P from TDT. These scripts are modified versions of example workbooks you can find on the TDT webpage (https://www.tdt.com/support/python-sdk/ ).

Events of recordings used in Figures 1, 2 and 5 were timestamped retrospectively in OpenScope (TDT) and FP signal was extracted around these timestamps with: 
SetupA_postrecording_epoc_importcam1.py

FED3-generated TTL pulses (https://open-ephys.org/fed3/fed3) during experiments in Figure 4 and 6 were used to timestamp nose pokes and pellet retrieval with a custom made gizmo (Thanks to Myles Billard) and FP signal was extracted with : 
SetupB_FEDpellet.py – for zscore around pellet retrieval
SetupB_FEDactive.py - for zscore around active nose pokes

Output traces of SetupB_FEDactive.py were separated into rewarded and non-rewarded nose pokes with:
FED3_PR_meanrewardedpoke_separation.py
(#Note: this code needs manual input in line 21/22 due to my rudimentary coding skills )

Finally, timing of nose pokes and pellet retrieval was extracted with: 
FIPHO_FEDtiming_pokeandpellet_setupB.py

This script also generates a trace of the complete recording with lines marking the nose pokes and pellets.
