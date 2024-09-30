Deep Learning for spectroscopic Data:

Application with (OSSL) Open Soil Spectral Library soil NIR/MIR Database: 
https://storage.googleapis.com/soilspec4gg-public/ossl_all_L1_v1.2.csv.gz

Load_dataset:
A dataset class for OSSL data adapted for torch dataloader.

base_net:
define two networks CuiNet and DeepSpectraCNN

utils:
training functions



CuiNet is derived from:
Modern practical convolutional neural networks for multivariate regression: Applications to NIR calibration
Chenhao Cui, Tom Fearn
[DOI: 10.1016/j.chemolab.2018.07.008 ](https://doi.org/10.1016/j.chemolab.2018.07.008)
2018


DeepSpectraCNN is derived from:
DeepSpectra: An end-to-end deep learning approach for quantitative spectral analysis
Xiaolei Zhang and Tao Lin and Jinfan Xu and Xuan Luo and Yibin Ying
https://doi.org/10.1016/j.aca.2019.01.002
2019

FullyConvNet is derived from:
Towards calibration-invariant spectroscopy using deep learning
M. Chatzidakis, G. A. Botton
https://doi.org/10.1038/s41598-019-38482-1
2019

ResNets 1D are adaptations in 1D and in dimensions of:
Deep Residual Learning for Image Recognition
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
https://doi.org/10.48550/arXiv.1512.03385
2015
