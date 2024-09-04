import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

class SpectralDataset(Dataset):
    def __init__(self, data_path, y_labels="oc.usda.c729", dataset_type="visnir", test_size=0.2, random_seed=42):
        self.data_path = data_path
        self.y_labels = y_labels
        self.dataset_type = dataset_type
        self.test_size = test_size
        self.random_seed =random_seed
        
        torch.manual_seed(self.random_seed)
         
      # Load data
        try:
            data_raw = pd.read_csv(self.data_path, low_memory=False)
        except FileNotFoundError:
            raise ValueError(f"File not found: {self.data_path}")
            
            
        # Handle y_labels parameter
        if isinstance(y_labels, str):
            y_labels = [y_labels]
        self.y_labels = y_labels
        
        missing_labels = [label for label in self.y_labels if label not in data_raw.columns]
        if missing_labels:
            raise ValueError(f"Missing labels in the dataset: {', '.join(missing_labels)}")
        
        # Extract target variables
        Y = np.array(data_raw.filter(regex="|".join(y_labels)))
        mask = ~np.isnan(Y).any(axis=1)
        
        # Initialize X and Y based on dataset type
        if dataset_type == "mir":
           X, Y = self._process_data(data_raw, mask, "mir", "abs")
        elif dataset_type == "nir":
           X, Y = self._process_data(data_raw, mask, "visnir", "ref")
        else:
           raise ValueError("dataset_type must be either 'mir' or 'nir'")
       
        self.spec_dims=X.shape[1]
        
         # Split the data into training and validation sets
        total_size = len(X)
        test_size = int(total_size * self.test_size)
        train_size = total_size - test_size
        
       # Generate indices
        indices = np.arange(total_size)
        
        # Set seed for reproducibility of split
        generator = torch.Generator().manual_seed(self.random_seed)
        shuffled_indices = torch.randperm(total_size, generator=generator).numpy()
        
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:]
        
        # Split data into train and validation sets
        self.X_train = torch.tensor(X[train_indices], dtype=torch.float32)
        self.Y_train = torch.tensor(Y[train_indices], dtype=torch.float32)
        self.X_val = torch.tensor(X[val_indices], dtype=torch.float32)
        self.Y_val = torch.tensor(Y[val_indices], dtype=torch.float32)
        
        
    def _process_data(self, data_raw, mask, spectrum_type, filter_keyword):
        # Extract spectrum data
        spectral_data = np.array(data_raw.filter(regex=spectrum_type).filter(regex=filter_keyword))
        mask_ = ~(np.isnan(spectral_data[:, 0])[mask])
        X = spectral_data[mask][mask_]
        Y = np.array(data_raw.filter(regex="|".join(self.y_labels)))[mask][mask_]
        return X , Y
        
    def __len__(self):
        return len(self.X)
    
    
    def get_spectral_dimensions(self):
        """
        Define the spectral dimension vector for x-axis labels based on dataset type.
        """
        if self.dataset_type == "nir":
            return np.linspace(350, 2500, self.spec_dims)  # For 'nir', dimensions between 350 and 2500
        elif self.dataset_type == "mir":
            return np.linspace(2500, 15384, self.spec_dims)  # For 'mir', dimensions between 2500 and 15384
        else:
            raise ValueError("dataset_type must be either 'nir' or 'mir'")