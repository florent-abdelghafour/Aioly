import os
from data.load_dataset_atonce import  SpectralDataset
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F

from net.chemtools.PLS import PLS
from net.chemtools.LWPLSR import LWPLSR

if __name__ == "__main__":
    
    user = os.environ.get('USERNAME')
    if user == 'fabdelghafo':
        data_path = "C:\\00_aioly\\GitHub\datasets\\ossl\\ossl_all_L1_v1.2.csv"
    else:
        data_path = "/home/metz/deepchemometrics/Aioly/data/dataset/oss/ossl_all_L1_v1.2.csv"

    y_labels = ["oc_usda.c729_w.pct", "na.ext_usda.a726_cmolc.kg"]
    dataset_type = "mir"

    # Instantiate the dataset
    start_time = time.time()
    dataset = SpectralDataset(data_path, y_labels, dataset_type)
    spec_dims=dataset.spec_dims
    end_time = time.time()
    loading_time = end_time - start_time
    print(f"Data loading time: {loading_time:.2f} seconds")
    
    
    wavelength = dataset.get_spectral_dimensions()
    num_samples = 200 
    X_train=dataset.X_train
    X_val=dataset.X_val
    Y_train=dataset.Y_train
    Y_val=dataset.Y_val

    train_size = len(dataset.X_train)
    train_sample_indices = np.random.choice(train_size, min(num_samples, train_size), replace=False)

    val_size = len(dataset.X_val)
    val_sample_indices = np.random.choice(val_size, min(num_samples, val_size), replace=False)
    
    
    X_train_sample = X_train[train_sample_indices].numpy()
    Y_train_sample = Y_train[train_sample_indices].numpy()

    X_val_sample = X_val[val_sample_indices].numpy()
    Y_val_sample = Y_val[val_sample_indices].numpy()
    
    epsilon = 1e-10
    Y_train_log = torch.log(torch.where(Y_train > 0, Y_train, torch.tensor(epsilon, dtype=Y_train.dtype)))
    Y_val_log = torch.log(torch.where(Y_val > 0, Y_val, torch.tensor(epsilon, dtype=Y_val.dtype)))
    
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    for i in range(min(num_samples, train_size)):
        plt.plot(wavelength,X_train_sample[i], alpha=0.5, label=f'Sample {i+1}' if i < 10 else "")
        plt.title("X_train")
        plt.xlabel("wavelenght (nm)")
        plt.ylabel("Pseudo absorbance")
        leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 fancybox=True, shadow=True, ncol=5, labelcolor=default_colors) 
    
    plt.subplot(2, 2, 3)
    for i in range(min(num_samples, val_size)):
         _=plt.plot(wavelength, X_val_sample[i], alpha=0.5, label=f'Sample {i+1}' if i < 10 else "")
    plt.title(" X_val")
    plt.xlabel("Wavelength nm")
    plt.ylabel("Pseudo absorbance")
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 fancybox=True, shadow=True, ncol=5, labelcolor=default_colors)
    
   
    plt.subplot(2, 2, 2)
    for j in range(len(y_labels)):
         _=plt.hist(Y_train_log[:, j], bins=100, alpha=0.5, label=y_labels[j])
    plt.title("Histogram of Sampled Training Targets (Y_train)")
    plt.xlabel("y value")
    plt.ylabel("Frequency")
    
    
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 fancybox=True, shadow=True, ncol=5, labelcolor=default_colors)
    
    plt.subplot(2, 2, 4)
    for j in range(len(y_labels)):
         _=plt.hist((Y_val_log[:, j]), bins=100, alpha=0.5, label=y_labels[j]);
    plt.title("Histogram of Sampled Validation Targets (Y_val)")
    plt.xlabel("y value")
    plt.ylabel("Frequency")
    leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 fancybox=True, shadow=True, ncol=5, labelcolor=default_colors)
    
    plt.tight_layout()
    plt.show(block=False)
    
    sample_size = int(0.25 * train_size)
    sample_indices = np.random.choice(train_size, sample_size, replace=False)
    X_sample = dataset.X_train[sample_indices].numpy()
    
    nb_pca_comp =4

    pca = PCA(n_components=nb_pca_comp)
    pca_scores = pca.fit(X_sample)
    pca_loadings =pca.components_.T*np.sqrt(pca.explained_variance_)
    
    plt.figure()
    for i in range(np.shape(pca_loadings)[1]):
        plt.plot(wavelength,pca_loadings[:,i],default_colors[i],label=f'PC{i+1}')
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("absorbance")  
        # lab= 'PC'+str(i+1)
        # plt.title(lab) 
        plt.title('Principal components')
        plt.grid()  
        plt.tight_layout()
        leg = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 fancybox=True, shadow=True, ncol=5, labelcolor=default_colors) 
    plt.show(block=False)
    
    
    
    
    
 
    
    
    
    
