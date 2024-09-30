Dataset Link :

https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/blob/master/notebooks/Tutorial_on_DL_optimization/datasets/mango_dm_full_outlier_removed2.mat


References Related to This Data

Full details about this version of the dataset can be found in the following article:
P. Mishra, D. Passos, A synergistic use of chemometrics and deep learning improved the predictive performance of near-infrared spectroscopy models for dry matter prediction in mango fruit, Chemometrics and Intelligent Laboratory Systems (2021), 104287
DOI: 10.1016/j.chemolab.2021.104287

The original dataset can be accessed here:
Original Dataset
N.T. Anderson, K.B. Walsh, P.P. Subedi, C.H. Hayes, Achieving robustness across season, location and cultivar for a NIRS model for intact mango fruit dry matter content, Postharvest Biology and Technology, 168 (2020), 111202
DOI: 10.1016/j.postharvbio.2020.111202

Acknowledgements

We would like to extend our sincere thanks to Dario Passos for his  contribution in preparing the data used in this tutorial. You can download the data directly from his GitHub at the following link: notebooks/Tutorial_on_DL_optimization/datasets/

This dataset includes a total of 11,691 NIR spectra (Near-Infrared, covering 684–990 nm with 3 nm sampling, resulting in 103 variables) and Dry Matter (DM%) measurements performed on 4,675 mango fruits across four harvest seasons: 2015, 2016, 2017, and 2018.

The data has been cleaned by removing a few outliers from the original dataset using Hotelling’s T² and Q statistics from a PLS (Partial Least Squares) decomposition. The version used in this tutorial is a concatenation of the original spectra along with different transformations, including SNV (Standard Normal Variate), 1st derivative, 2nd derivative, SNV + 1st derivative, and SNV + 2nd derivative.

File Details

    File Name: mango_dm_full_outlier_removed2.mat
    File Type: MATLAB
