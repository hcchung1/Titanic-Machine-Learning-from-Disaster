import torch # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from torch.utils.data import Dataset # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import numpy as np # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from PIL import Image # pyright: ignore[reportMissingModuleSource, reportMissingImports]


# ========== Global Configurations ==========
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

CUR_MODEL = 'BaseCNN'
KAGGLE_SUBMIT = True
EARLY_STOPPING = True
MULTI_GPU = False
NUM_WORKERS = 0
BATCH_NORM = False
RESIDUAL = False
DROPOUT = 0
DROPOUT_ENABLED = False
POOLING = 'max'  # 'avg', 'max', 'str2conv', 'none'
POOLING_ENABLED = True 
# ===========================================

class FashionMNISTCSVDataset(Dataset):
    """Custom Dataset for loading Fashion MNIST from CSV files"""
    def __init__(self, csv_file, transform=None, has_idx=False, is_test=False):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.has_idx = has_idx
        self.is_test = is_test
        
        if not self.is_test:
            self.labels = self.data_frame['label'].values
        else:
            self.labels = None  # validation and test datasets no need for labels

        # read image data from pixel1 to pixel784
        pixel_columns = [f'pixel{i}' for i in range(1, 785)]
        self.images = self.data_frame[pixel_columns].values
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        
        image = self.images[idx].reshape(28, 28).astype(np.uint8)
        image = Image.fromarray(image, mode='L')
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            if self.has_idx and 'idx' in self.data_frame.columns:
                sample_idx = int(self.data_frame.iloc[idx]['idx'])
            else:
                sample_idx = int(idx)
            return image, sample_idx
        else:
            label = int(self.labels[idx])
            return image, label
        