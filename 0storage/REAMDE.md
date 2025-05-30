0model_weight includde:
  pretrained weights of PSS-Finder

1select_gomc includes:
  the original gomc data (10,000 from the previous original data)

2select_gomc_2 includes:
  the output of classification model, which is just the result of the input gomc data

3select_gomc_3 includes:
  the output of input gomc data, which is splited by its scaffold type

4select_gomc_4 includes:
  the structure predicted from the ESMFold

5select_gomc_5 includes:
  the structure of SBP from SYNBIP database. it's the reference structure for new structure to align.

6select_gomc_6 includes:
 the final output of the PSS-Finder, the final file is with TM-Score and RMSD.

PP.xlsx is a reference file for intial screening. it contains some basic scaffold information to exclude obviously wrong judgemente.

