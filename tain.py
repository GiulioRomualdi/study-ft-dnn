from ft_denoising.dataset import FTDataset
from ft_denoising.network import SimpleFTDenoising
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def define_model():
    return SimpleFTDenoising(
        layers=4,
        in_channel=[7, 100, 100, 100],
        out_channel=[100, 100, 100, 100],
    )


# Create the dataset
def get_dataset(device):
    training_data = FTDataset(
            dataset_path_folder=Path("/home/gromualdi/robot-code/test-ft-dnn/datasets/r_leg_ft_calib_nn.mat"),
            ft_name="r_leg_ft",
            device=device,
        )

    validation_data = FTDataset(
            dataset_path_folder=Path("/home/gromualdi/robot-code/test-ft-dnn/datasets/r_leg_ft_valid_nn.mat"),
            ft_name="r_leg_ft",
            device=device,
            max_raw_ft=training_data.max_raw_ft,
            max_expected=training_data.max_expected,
            max_temperature=training_data.max_temperature,
        )


    train_dataloader = DataLoader(training_data, batch_size=256, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=len(validation_data), shuffle=False)

    return train_dataloader, validation_dataloader

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    model = define_model().to(device)


    # suggest optimizer for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_dataloader, validation_dataloader = get_dataset(device)

    loss = torch.nn.MSELoss()
    loss_validation = torch.nn.MSELoss()

    writer = SummaryWriter()
    for epoch in range(1000):
        model.train()

        total_loss = 0.0

        for i, data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            ft_raw = inputs['ft_raw']
            temperature = inputs['temperature']

            # concatenate the inputs
            ft_raw_input = torch.cat((ft_raw, temperature), dim=1)

            expected = labels['expected']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(ft_raw_input)
            loss_value = loss(outputs, expected)
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item() 

        # validate the model
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(validation_dataloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                ft_raw = inputs['ft_raw']
                temperature = inputs['temperature']

                # concatenate the inputs
                ft_raw_input = torch.cat((ft_raw, temperature), dim=1)

                expected = labels['expected']

                # forward + backward + optimize
                model.eval()
                outputs = model(ft_raw_input)
                loss_value = loss_validation(outputs, expected)

                validation_loss += loss_value.item()

            # print statistics
            print(f"Epoch: {epoch}, Validation Loss: {validation_loss} , Train Loss: {total_loss}")
            writer.add_scalar("Loss/training", total_loss, epoch)
            writer.add_scalar("Loss/validation", validation_loss, epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), "model" + str(epoch) + ".pth")



if __name__ == "__main__":

    train()