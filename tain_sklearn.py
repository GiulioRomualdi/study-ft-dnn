from ft_denoising.dataset import FTDataset
import joblib


from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error



# Create the dataset
def get_dataset(device="cpu"):
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

    # load dataset in numpy
    training_data_raw_ft = training_data.raw_ft.cpu().numpy()
    training_data_temperature = training_data.temperature.cpu().numpy()
    training_data_expected = training_data.expected.cpu().numpy()

    # stack the data together
    X_training = np.hstack((training_data_raw_ft, training_data_temperature))
    y_training = training_data_expected

    validation_data_raw_ft = validation_data.raw_ft.cpu().numpy()
    validation_data_temperature = validation_data.temperature.cpu().numpy()
    validation_data_expected = validation_data.expected.cpu().numpy()

    # stack the data together
    X_validation = np.hstack((validation_data_raw_ft, validation_data_temperature))
    y_validation = validation_data_expected


    return X_training, y_training, X_validation, y_validation

def train():
    X_train, Y_train, X_val, Y_val = get_dataset()


    # perform crossvalidation of the hyperparameters
    param_grid = {
        'alpha': [0.01, 0.1, 1.0],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': [0.01, 0.1, 1.0]
    }

    kr = KernelRidge()
    grid_search = GridSearchCV(kr, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=5, )


    # Fit the model using the training data
    grid_search.fit(X_train, Y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Print the best hyperparameters
    print("Best Hyperparameters:", best_params)

    # Use the best model to make predictions on the validation set
    Y_pred = grid_search.best_estimator_.predict(X_val)

    # Evaluate the model on the validation set
    mse = mean_squared_error(Y_val, Y_pred)
    print("Mean Squared Error on Test Set:", mse)


    # save the model
    joblib.dump(grid_search.best_estimator_, 'model.pkl')

if __name__ == "__main__":

    train()
