import numpy as np
import pandas as pd



def process_data(data_raw, mask, spectrum_type, filter_keyword,y_labels):
    # Extract spectrum data
    spectral_data = np.array(data_raw.filter(regex=spectrum_type).filter(regex=filter_keyword))
    mask_ = ~(np.isnan(spectral_data[:, 0])[mask])
    X = spectral_data[mask][mask_]
    Y = np.array(data_raw.filter(regex="|".join(y_labels)))[mask][mask_]
    return X , Y


dataset_type="mir"

data_path="C:\\00_aioly\\sources_projects\\OSSL_project\\data\\datasets\\ossl\\ossl_all_L1_v1.2.csv"

data_raw = pd.read_csv(data_path, low_memory=False)
y_names = data_raw.columns


y_labels=["oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg"] #


Y = np.array(data_raw.filter(regex="|".join(y_labels)))
mask = ~np.isnan(Y).any(axis=1)


if dataset_type == "mir":
    X, Y = process_data(data_raw, mask, "mir", "abs",y_labels)
elif dataset_type == "nir":
    X, Y = process_data(data_raw, mask, "visnir", "ref",y_labels)
    
 