import numpy as np
from docutils.parsers.rst.directives.misc import Class


def snv(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return output_data

## data augmentation of NIR data, add slope, offset, noise and shift
class data_augmentation:
    def __init__(self, slope = 0.1, offset = 0.1, noise = 0.1, shift = 0.1):
        self.slope = slope
        self.offset = offset
        self.noise = noise
        self.shift = shift

    def __call__(self, X):
        slope_factor = np.random.uniform(1 - self.slope, 1 + self.slope, X.shape)
        offset_factor = np.random.uniform(-self.offset, self.offset, X.shape)
        noise_factor = np.random.normal(0, self.noise, X.shape)
        X_aug = X * slope_factor + offset_factor + noise_factor
        return X_aug