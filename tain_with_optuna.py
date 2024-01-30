from ft_denoising.dataset import FTDataset
from ft_denoising.network import SimpleFTDenoising
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import optuna
import os
import shutil
from optuna.storages import RetryFailedTrialCallback

CHECKPOINT_DIR = "pytorch_checkpoint"


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = []
    in_channel = []
    in_features = 7
    for i in range(n_layers):
        out_channel = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_channel))
        layers.append(nn.ReLU())

        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_channel

    layers.append(nn.Linear(in_features, 6))

    return nn.Sequential(*layers)

# Create the dataset
def get_dataset(device):
    training_data = FTDataset(
            dataset_path_folder=Path("/home/gromualdi/robot-code/test-ft-dnn/datasets/r_leg_ft_valid_nn.mat"),
            ft_name="r_leg_ft",
            device=device,
        )

    validation_data = FTDataset(
            dataset_path_folder=Path("/home/gromualdi/robot-code/test-ft-dnn/datasets/r_leg_ft_calib_nn.mat"),
            ft_name="r_leg_ft",
            device=device,
            max_raw_ft=training_data.max_raw_ft,
            max_expected=training_data.max_expected,
            max_temperature=training_data.max_temperature,
        )


    train_dataloader = DataLoader(training_data, batch_size=256, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=len(validation_data), shuffle=False)


    return train_dataloader, validation_dataloader

def objective(trial):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    model = define_model(trial).to(device)

    trial_number = RetryFailedTrialCallback.retried_trial_number(trial)
    trial_checkpoint_dir = os.path.join(CHECKPOINT_DIR, str(trial_number))
    checkpoint_path = os.path.join(trial_checkpoint_dir, "model.pt")
    checkpoint_exists = os.path.isfile(checkpoint_path)

    if trial_number is not None and checkpoint_exists:
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint["epoch"]
        epoch_begin = epoch + 1

        print(f"Loading a checkpoint from trial {trial_number} in epoch {epoch}.")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        accuracy = checkpoint["accuracy"]
    else:
        trial_checkpoint_dir = os.path.join(CHECKPOINT_DIR, str(trial.number))
        checkpoint_path = os.path.join(trial_checkpoint_dir, "model.pt")
        epoch_begin = 0

    os.makedirs(trial_checkpoint_dir, exist_ok=True)
    # A checkpoint may be corrupted when the process is killed during `torch.save`.
    # Reduce the risk by first calling `torch.save` to a temporary file, then copy.
    tmp_checkpoint_path = os.path.join(trial_checkpoint_dir, "tmp_model.pt")

    print(f"Checkpoint path for trial is '{checkpoint_path}'.")


    # suggest optimizer for regression
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataloader, validation_dataloader = get_dataset(device)

    loss = torch.nn.MSELoss()
    loss_validation = torch.nn.MSELoss()

    for epoch in range(100):
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
            # print(f"Epoch: {epoch}, Validation Loss: {validation_loss} , Train Loss: {total_loss}")
            # writer.add_scalar("Loss/training", total_loss, epoch)
            # writer.add_scalar("Loss/validation", validation_loss, epoch)

            trial.report(validation_loss, epoch)


            print(f"Saving a checkpoint in epoch {epoch}.")

            torch.save(
                {
                  "epoch": epoch,
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "accuracy": validation_loss,
                },
            tmp_checkpoint_path,
            )
            shutil.move(tmp_checkpoint_path, checkpoint_path)


        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return validation_loss


if __name__ == "__main__":

    storage = optuna.storages.RDBStorage(
        "sqlite:///example.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    
    study = optuna.create_study(
        storage=storage, study_name="pytorch_checkpoint", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=10, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # The line of the resumed trial's intermediate values begins with the restarted epoch.
    optuna.visualization.plot_intermediate_values(study).show()
