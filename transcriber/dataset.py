from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class DacDataset(Dataset):
    def __init__(self, data_path, labels_path):
        data_path = Path(data_path) if not isinstance(Path, data_path) else data_path
        labels_path = Path(labels_path) if not isinstance(Path, labels_path) else labels_path
        assert data_path.exists(), f"Data path {data_path} does not exist!"
        assert labels_path.exists(), f"Labels path {labels_path} does not exist!"
        self.data_files = list(sorted(data_path.glob('*.pt')))
        self.label_files = list(sorted(labels_path.glob('*.npz')))

    def __getitem__(self, index):
        """Function to return one feature-label pair from the dataset.

        Parameters
        ----------
        index (int): index to sample items

        Returns
        -------
        dac features (torch Tensor)
        label (torch Tensor)
        """
        data = torch.load(self.data_files[index], map_location='cpu')['latent_space']
        # find label corresponding to data file
        label_file = [self.label_files[i] for i, f in enumerate(self.label_files) if f.stem == self.data_files[index].stem][0]
        label = torch.from_numpy(np.load(label_file)['pitch_label'])
        return data.type(torch.FloatTensor), label.type(torch.FloatTensor)

    def __len__(self):
        return len(self.data_files)