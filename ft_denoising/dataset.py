import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import h5py
from typing import List


class FTDataset(Dataset):
    def __init__(
        self,
        dataset_path_folder: Path,
        ft_name: str,
        device=torch.device("cpu"),
        max_raw_ft: float = None,
        max_expected: float = None,
        max_temperature: float = None,
    ):
      
        self.device = device

        # open the file with h5py
        print("Opening file: ", dataset_path_folder)
        with h5py.File(str(dataset_path_folder), "r") as file:

            # read the data
            self.raw_ft = file[ft_name]['data'][:].T
            self.raw_ft = torch.tensor(self.raw_ft, dtype=torch.float32, device=self.device)

            # normalize the data by getting the max value in each column
            if max_raw_ft is None:
                self.max_raw_ft = torch.max(torch.abs(self.raw_ft), dim=0).values
            else:
                self.max_raw_ft = max_raw_ft   
            
            self.raw_ft = self.raw_ft / self.max_raw_ft

            
            self.expected = file[ft_name]['expected'][:].T
            self.expected = torch.tensor(self.expected, dtype=torch.float32, device=self.device)

            # normalize the data by getting the max value in each column
            if max_expected is None:
                self.max_expected = torch.max(torch.abs(self.expected), dim=0).values
            else:
                self.max_expected = max_expected

            self.expected = self.expected / self.max_expected

            self.temperature = file[ft_name]['temperature'][:].T
            self.temperature = torch.tensor(self.temperature, dtype=torch.float32, device=self.device)

            # normalize the data
            if max_temperature is None:
                self.max_temperature = torch.max(torch.abs(self.temperature))
            else:
                self.max_temperature = max_temperature

            self.temperature = self.temperature / self.max_temperature

    def __getitem__(self, index):
        return {
            "ft_raw": self.raw_ft[index,:],
            "temperature": self.temperature[index,:],
        }, {
            "expected": self.expected[index,:],
        }

    def __len__(self):
        return self.raw_ft.shape[0]


if __name__ == "__main__":
    training_data = FTDataset(
        dataset_path_folder=Path("/home/gromualdi/robot-code/test-ft-dnn/datasets/r_leg_ft_calib_nn.mat"),
        ft_name="r_leg_ft",
    )
    print(training_data[0])

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(training_data, batch_size=5, shuffle=False)
    rain_features, train_labels = next(iter(train_dataloader))
    # print(
    #     f"Feature batch shape: {train_features['data'].size()}"
    # )  # batch_size, sequence_length, 6

    # print(f"Lables batch shape: {train_features['dataset_index'].size()}")  # batch_size

    # print(f"Lables batch shape: {train_labels['pose'].size()}")  # batch_size, 4, 4
    # print(
    #     f"Lables batch shape: {train_labels['spatial_velocity'].size()}"
    # )  # batch_size, 3

    # train_features, train_labels = training_data[0]

    # print(train_features["dataset_index"])