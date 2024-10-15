from train_model import load_data, min_max_scale
import sys
import numpy as np
from typing import Tuple
from yaml import load, FullLoader

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
MAGENTA = "\033[35m"
NC = "\033[0m"


def load_thetas(path: str) -> Tuple[float, float]:
    '''
    [LOAD_THETAS] Loads theta0 and theta1 values from a YAML file,
    handling potential errors.
    '''
    try:
        with open(path, 'r') as file:
            thetas = load(file, Loader=FullLoader)
            if thetas is None or 'theta0' not in thetas or \
                    'theta1' not in thetas:
                raise KeyError("thetas.yaml is missing the required values")
        theta0 = float(thetas['theta0'])
        theta1 = float(thetas['theta1'])

    except ValueError:
        print(f"\n  {RED}ValueError:{NC} thetas.yaml contains non-numeric "
              "values\n")
        sys.exit(1)
    except KeyError as e:
        print(f"\n  {RED}KeyError:{NC} {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n  {RED}An unexpected error occured:{NC} {e}\n")
        sys.exit(1)

    return theta0, theta1


def calculate_precision(predicted: np.ndarray, actual: np.ndarray, m: float) -> None:
    '''
    [CALCULATE_PRECISION] Computes and prints MAE, MSE, and RMSE to
    evaluate the precision of the model's predictions.
    '''
    print(f"  {MAGENTA}Values statistics:{NC}")
    pred_min, actual_min = np.min(predicted), np.min(actual)
    pred_max, actual_max = np.max(predicted), np.max(actual)
    pred_mean, actual_mean = np.mean(predicted), np.mean(actual)
    print(f"   - minimum: predicted = {pred_min:.3f}, actual = {actual_min:.3f}")
    print(f"   - maximum: predicted = {pred_max:.3f},  actual = {actual_max:.3f}")
    print(f"   - mean:    predicted = {pred_mean:.3f},  actual = {actual_mean:.3f}\n")

    mae = np.sum(np.abs(predicted - actual)) / m
    mse = np.sum((predicted - actual) ** 2) / m
    rmse = np.sqrt(mse)

    print(f"  {GREEN}Mean Absolute Error (MAE):{NC} {mae:.3f}")
    print(f"   - predictions are off by about {mae * 100:.3f}% of the total range\n")
    print(f"  {GREEN}Mean Squared Error (MSE):{NC} {mse:.3f}")
    print(f"   - average of the squares of the errors\n")
    print(f"  {GREEN}Root Mean Squared Error (RMSE):{NC} {rmse:.3f}")
    print(f"   - average scaled distance of predictions from actual values\n")


def print_start(m: float, theta0: float, theta1: float) -> None:
    '''
    [PRINT_START] Prints the number of samples and the parameters
    (theta0, theta1) used in the model before calculating precision.
    '''
    print(f"\n  {YELLOW}Calculating model precision...{NC}\n")
    print(f"    samples: {m} | theta0: {theta0:.3f} | theta1: {theta1:.3f}\n")


def main():
    '''
    [MAIN] Loads the data, parameters, and scales the datasets
    before calculating predictions and evaluating model precision.
    '''
    mileage_set, price_set = load_data("data.csv")
    theta0, theta1 = load_thetas("thetas.yaml")
    rs_mileage, rs_price = min_max_scale(mileage_set, price_set)
    prediction = theta0 + (theta1 * rs_mileage)
    m = len(rs_price)
    print_start(m, theta0, theta1)
    calculate_precision(prediction, rs_price, m)

if __name__ == "__main__":
    main()
