import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import h5py
from typing import List

class FTDataset(Dataset):
    def __init__(
        self,
        datasets_path_folder: Path,
        ft_name: str,
        device=torch.device("cpu"),
        max_raw_ft: torch.Tensor = None,
        max_temperature: torch.Tensor = None,
        max_accelerometer: torch.Tensor = None,
        max_gyro: torch.Tensor = None,
        max_expected: torch.Tensor = None,
    ):      
        self.device = device

        # get all the mat files in the folder
        self.datasets_path_folder = datasets_path_folder
        self.datasets_path_folder = self.datasets_path_folder.glob("*.mat")
        self.datasets_path_folder = list(self.datasets_path_folder)

        self.raw_ft = torch.empty(0, 6, device=self.device)
        self.temperature = torch.empty(0, 1, device=self.device)
        self.accelerometer = torch.empty(0, 3, device=self.device)
        self.gyro = torch.empty(0, 3, device=self.device)
        self.expected = torch.empty(0, 6, device=self.device)

        for dataset_path_folder in self.datasets_path_folder:
            raw_ft, gyro, accelerometer, temperature, expected = self._load_one_dataset(dataset_path_folder, ft_name)

            self.raw_ft = torch.cat((self.raw_ft, raw_ft), 0)
            self.temperature = torch.cat((self.temperature, temperature), 0)
            self.accelerometer = torch.cat((self.accelerometer, accelerometer), 0)
            self.gyro = torch.cat((self.gyro, gyro), 0)
            self.expected = torch.cat((self.expected, expected), 0)

        # normalize the data
        if max_raw_ft is None:
            self.max_raw_ft = torch.max(torch.abs(self.raw_ft), dim=0).values
        else:
            self.max_raw_ft = max_raw_ft
        self.raw_ft = self.raw_ft / self.max_raw_ft

        if max_temperature is None:
            self.max_temperature = torch.max(torch.abs(self.temperature), dim=0).values
        else:
            self.max_temperature = max_temperature
        self.temperature = self.temperature / self.max_temperature

        if max_accelerometer is None:
            self.max_accelerometer = torch.max(torch.abs(self.accelerometer), dim=0).values
        else:
            self.max_accelerometer = max_accelerometer

        self.accelerometer = self.accelerometer / self.max_accelerometer

        if max_gyro is None:
            self.max_gyro = torch.max(torch.abs(self.gyro), dim=0).values
        else:
            self.max_gyro = max_gyro

        self.gyro = self.gyro / self.max_gyro

        if max_expected is None:
            self.max_expected = torch.max(torch.abs(self.expected), dim=0).values
        else:
            self.max_expected = max_expected

        self.expected = self.expected / self.max_expected

        print(self.expected.shape)


    def __getitem__(self, index):
        return {
            "ft_raw": self.raw_ft[index,:],
            "gyro": self.gyro[index,:],
            "accelerometer": self.accelerometer[index,:],
            "temperature": self.temperature[index,:],
        }, {
            "expected": self.expected[index,:],
        }

    def __len__(self):
        return self.raw_ft.shape[0]

    def _load_one_dataset(self, dataset_path_folder, ft_name):
        # open the file with h5py
        print("Opening file: ", dataset_path_folder)
        with h5py.File(str(dataset_path_folder), "r") as file:
            file = file["parsedFile"]
            file_measures = file["measures"]

            # read the data
            raw_ft = file_measures["FTs"][ft_name]['data'][:].T
            raw_ft = torch.tensor(raw_ft, dtype=torch.float32, device=self.device)
            
            gyro = file_measures["gyros"][ft_name]['data'][:].T
            gyro = torch.tensor(gyro, dtype=torch.float32, device=self.device)

            accelerometer = file_measures["accelerometers"][ft_name]['data'][:].T
            accelerometer = torch.tensor(accelerometer, dtype=torch.float32, device=self.device)

            temperature = file_measures["temp"][ft_name]['data'][:].T
            temperature = torch.tensor(temperature, dtype=torch.float32, device=self.device)

            expected = file["expectedValues"]["FTs"][ft_name]['data'][:].T
            expected = torch.tensor(expected, dtype=torch.float32, device=self.device)

            return raw_ft, gyro, accelerometer, temperature, expected


if __name__ == "__main__":
    training_data = FTDataset(
        datasets_path_folder=Path("/home/gromualdi/robot-code/element_ft-nonlinear-modeling/code/matlab/src/training"),
        ft_name="r_leg_ft",
    )
    print(training_data[0])

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(training_data, batch_size=5, shuffle=False)
    train_features, train_labels = next(iter(train_dataloader))
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