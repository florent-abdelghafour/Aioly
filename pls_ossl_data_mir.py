import os
from data.load_dataset_atonce import  SpectralDataset
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F

from net.chemtools.PLS import PLS

import pickle

def ccc(y_true, y_pred):
    # Calculate correlation
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    cor = torch.mean((y_true - mean_true) * (y_pred - mean_pred)) / (torch.std(y_true) * torch.std(y_pred))

    # Population variances
    var_true = torch.var(y_true, unbiased=False)
    var_pred = torch.var(y_pred, unbiased=False)

    # Population standard deviations
    sd_true = torch.std(y_true, unbiased=False)
    sd_pred = torch.std(y_pred, unbiased=False)

    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    ccc = numerator / denominator

    return ccc.item()

def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()

if __name__ == "__main__":
    
    #########################################################################################################
    user = os.environ.get('USERNAME')
    if user == 'fabdelghafo':
        data_path = "C:\\00_aioly\\GitHub\datasets\\ossl\\ossl_all_L1_v1.2.csv"
    else:
        data_path = "/home/metz/deepchemometrics/Aioly/data/dataset/oss/ossl_all_L1_v1.2.csv"

    y_labels = ["oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg","clay.tot_usda.a334_w.pct","k.ext_usda.a725_cmolc.kg","ph.h2o_usda.a268_index"]  
    dataset_type = "nir"

    # Instantiate the dataset
    start_time = time.time()
    dataset = SpectralDataset(data_path, y_labels, dataset_type)
    spec_dims=dataset.spec_dims
    end_time = time.time()
    loading_time = end_time - start_time
    print(f"Data loading time: {loading_time:.2f} seconds")
    #########################################################################################################
    
    wavelength = dataset.get_spectral_dimensions()
    num_samples = 200 
    X_train=dataset.X_train
    X_val=dataset.X_val
    Y_train=dataset.Y_train
    Y_val=dataset.Y_val
    #########################################################################################################
    ncomp=50
    pls =PLS(ncomp=ncomp)
    #########################################################################################################

    
    base_path = 'C:\\00_aioly\\GitHub\\datasets\\ossl\\figures\\pls\\visnir'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams['font.family'] = 'Times New Roman'
    
    for target_label in y_labels:
        target_index = y_labels.index(target_label)
        Y_train_subset = Y_train[:, target_index].unsqueeze(1)
        Y_val_subset = Y_val[:, target_index].unsqueeze(1)

        Y_train_subset = torch.log1p(Y_train_subset)
        Y_val_subset = torch.log1p(Y_val_subset)

        pls.fit(X_train, Y_train_subset)

        perf = []
        for lv in range(ncomp):
            y_pred = pls.predict(X_val, lv)
            rmse = torch.sqrt(F.mse_loss(y_pred, Y_val_subset, reduction='none')).mean(dim=0)
            perf.append(rmse)

        y_pred_final = pls.predict(X_val, ncomp - 1)  
        ccc_value = ccc(Y_val_subset, y_pred_final)
        r2_value = r2_score(Y_val_subset, y_pred_final)
        
        fig=plt.figure()
        plt.plot(range(1, ncomp + 1), [p.item() for p in perf], label=target_label,color =default_colors[target_index])
        
        plt.text(0.95, 0.95, f'CCC: {ccc_value:.2f}\nR²: {r2_value:.2f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
             color='red', fontweight='bold', fontfamily='serif')
        
        plt.xlabel('Latent Variables')
        plt.ylabel('RMSEP')
        plt.title(f'Training RMSE for {target_label} (log x +1)')
        plt.tight_layout()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True, ncol=5,labelcolor=default_colors[target_index],fontsize=12)
        plt.subplots_adjust(bottom=0.3)
        plt.grid(True)
        fig.show()
       
       
        pdf_path = os.path.join(base_path, f'fig_RMSE_{target_label}.pdf')
        plt.savefig(pdf_path, format='pdf')

        pickle_path = os.path.join(base_path, f'fig_RMSE_{target_label}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)
              
        fig, ax = plt.subplots()
        hexbin = ax.hexbin(Y_val_subset.squeeze().numpy(), y_pred_final.squeeze().numpy(), gridsize=50, cmap='viridis', mincnt=1)
        cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
        cb.set_label('Density')
        lims = [np.min([Y_val_subset.numpy(), y_pred_final.numpy()]), np.max([Y_val_subset.numpy(), y_pred_final.numpy()])]
        ax.plot(lims, lims, 'k-', label= target_label)  
        
        plt.text(0.05, 0.95, f'CCC: {ccc_value:.2f}\nR²: {r2_value:.2f}', 
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                 color='red', fontweight='bold', fontfamily='serif')
        
        ax.set_xlabel('Real Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Predicted vs Real Values for {target_label} (log x + 1)')
        
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   fancybox=True, shadow=True, ncol=5, fontsize=12,labelcolor =default_colors[target_index])
        plt.tight_layout()
        plt.grid()
        fig.show()
        
        hexbin_pdf_path = os.path.join(base_path, f'fig_hexbin_{target_label}.pdf')
        plt.savefig(hexbin_pdf_path, format='pdf')

        # Save the figure object using pickle
        hexbin_pickle_path = os.path.join(base_path, f'fig_hexbin_{target_label}.pkl')
        with open(hexbin_pickle_path, 'wb') as f:
            pickle.dump(fig, f)