import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SoilSpectralDataSet(Dataset):
    def __init__(self, dataset_type="visnir", data_path=None, preprocessing=None,y_labels="oc.usda.c729", reduce_lbd = False):
        if data_path==None:
            # Set default data path in project path if none provided
            rel_dir = os.getcwd()
            data_path = os.path.join(rel_dir, 'data/dataset/oss/ossl_all_L1_v1.2.csv')
            
        self.reduce_lbd = reduce_lbd
        self.data_path = data_path
        self.oss_data_path = data_path
        self.preprocessing =preprocessing

      
        
       # Load data
        try:
            data_raw = pd.read_csv(self.data_path, low_memory=False)
        except FileNotFoundError:
            raise ValueError(f"File not found: {self.data_path}")
            
            
        # Handle y_labels parameter
        if isinstance(y_labels, str):
            y_labels = [y_labels]
        self.y_labels = y_labels
        
        # Extract target variables
        Y = np.array(data_raw.filter(regex="|".join(y_labels)))
        mask = ~np.isnan(Y).any(axis=1)
        
        # Initialize X and Y based on dataset type
        if dataset_type == "mir":
           self.X, self.Y = self._process_data(data_raw, mask, "mir", "abs")
        elif dataset_type == "nir":
           self.X, self.Y = self._process_data(data_raw, mask, "visnir", "ref")
        else:
           raise ValueError("dataset_type must be either 'mir' or 'nir'")
       
        self.spec_dims=self.X.shape[1]
        if self.reduce_lbd : 
            self.spec_dims -= 1
        
        # Apply preprocessing if provided
        if self.preprocessing:
           self.X = self.preprocessing(self.X)
        

    def _process_data(self, data_raw, mask, spectrum_type, filter_keyword):
        # Extract spectrum data
        spectral_data = np.array(data_raw.filter(regex=spectrum_type).filter(regex=filter_keyword))
        mask_ = ~(np.isnan(spectral_data[:, 0])[mask])
        X = spectral_data[mask][mask_]
        Y = np.array(data_raw.filter(regex="|".join(self.y_labels)))[mask][mask_]
        return X , Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        spectral_data = self.X[index, :]
        if self.reduce_lbd : 
            spectral_data = spectral_data[:-1]
        label = self.Y[index]
        return spectral_data, np.log(label+1)
    
    def get_labels(self):
        return(self.y_names)




# if __name__ == '__main__':


#     spectral_data = SoilSpectralDataSet("mir")
#     spectral_loader = DataLoader(spectral_data, shuffle=True, batch_size=32, num_workers=0)


#     for X, Y in spectral_loader:
#         print(X.shape)
#         print(Y.shape)
#         plt.plot(X.T)
#         break