import sys
import os
import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from yaml import dump

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
MAGENTA = "\033[35m"
NC = "\033[0m"

options = {
    'learn_rate': 0.15,
    'iters': 500,
    'plot_learning': False,
    'plot_end': False,
    'plot_input': False,
    'set_learn_values': False
}


def export_thetas(theta0: float, theta1: float) -> None:
    '''
    [EXPORT_THETAS] Exports theta0 and theta1 values to a YAML file.
    '''
    data = {
        'theta0': float(theta0),
        'theta1': float(theta1)
    }
    with open('thetas.yaml', 'w') as file:
        dump(data, file)
    print(f"\n  {GREEN}Exporting theta values into thetas.yaml{NC}\n")


def print_start(m: float, learn_rate: float, iters: float) -> None:
    '''
    [PRINT_START] Prints the initial parameters before training the model.
    '''
    print(f"\n  {YELLOW}Starting linear regression model training...{NC}\n")
    print(f"    samples: {m} | learn_rate: {learn_rate} | iterations: {iters}")


def train_regression(mileage: np.ndarray, price: np.ndarray) \
        -> Tuple[float, float]:
    '''
    [TRAIN_REGRESSION] Trains a linear regression model using gradient
    descent to find theta0 and theta1.
    '''
    learn_rate, iters = options['learn_rate'], options['iters']
    m, theta0, theta1 = len(mileage), 0, 0
    print_start(m, learn_rate, iters)
    for i in range(iters):
        prediction = theta0 + (theta1 * mileage)
        abs_err = prediction - price
        corr_grad0 = learn_rate * (1/m) * np.sum(abs_err)
        corr_grad1 = learn_rate * (1/m) * np.sum(abs_err * mileage)
        theta0 -= corr_grad0
        theta1 -= corr_grad1
        if (options['plot_learning']):
            plot_data(mileage, price, theta0, theta1, True)
    return theta0, theta1


def plot_data(mileage: np.ndarray, price: np.ndarray, theta0: float = 0,
              theta1: float = 0, live: bool = False) -> None:
    '''
    [PLOT_DATA] Plots the data points of mileage and price, with an optional
    regression line based on the presence of theta0 and theta1.
    '''
    plt.scatter(mileage, price, marker='o', alpha=0.8, c='orange')
    if theta0 and theta1:
        plt.plot(mileage, theta0 + (theta1 * mileage), c='teal')
    plt.gcf().canvas.manager.set_window_title("ft_linear_regression")
    plt.title('relationship between mileage and car price', pad=10)
    plt.xlabel('feature [mileage]', labelpad=10)
    plt.ylabel('target [price]', labelpad=10)
    if live:
        plt.pause(0.01)
        plt.clf()
    else:
        plt.show()


def min_max_scale(mileage: np.ndarray, price: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    '''
    [MIN_MAX_SCALE] Applies min-max scaling to mileage and price arrays.
    This prevents theta1 being updated more for the feature than for the
    target.
    '''
    feature_min, target_min = np.min(mileage), np.min(price)
    feature_max, target_max = np.max(mileage), np.max(price)

    mileage = (mileage - feature_min) / (feature_max - feature_min)
    price = (price - target_min) / (target_max - target_min)
    return mileage, price


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    '''
    [LOAD_DATA] Loads a dataset from a CSV file and returns mileage
    and price values as NumPy arrays.
    '''
    try:
        if path[-4:] != ".csv":
            raise AssertionError(f"file '{path}' not supported")
        if not os.path.exists(path):
            raise AssertionError(f"'{path}': no such file")
        dataset = pd.read_csv(path)
        data = dataset.to_numpy()

    except AssertionError as e:
        print(f"\n  {RED}AssertionError:{NC} {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  {RED}An unexpected error occured:{NC} {e}")
        sys.exit(1)

    data_size = len(data)
    feature = np.array([0] * data_size, float)
    target = np.array([1] * data_size, float)
    for i in range(data_size):
        feature[i] = data[i][0]
        target[i] = data[i][1]
    return feature, target


def main():
    '''
    [MAIN] Loads data, scales it, trains a regression model, and exports the
    resulting parameters into a YAML file.
    '''
    try:
        mileage_set, price_set = load_data("data.csv")
        if options['plot_input']:
            plot_data(mileage_set, price_set)
            sys.exit(0)
        rs_mileage, rs_price = min_max_scale(mileage_set, price_set)
        theta0, theta1 = train_regression(rs_mileage, rs_price)
        export_thetas(theta0, theta1)
        if options['plot_end']:
            plot_data(rs_mileage, rs_price, theta0, theta1)

    except ValueError as e:
        print(f"\n  {RED}ValueError:{NC} {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n  {RED}An unexpected error occured:{NC} {e}\n")
        sys.exit(1)


def handle_interrupt(signum, frame):
    '''
    [HANDLE_INTERRUPT] Signal handler for SIGINT (Ctrl+C).
    '''
    print(f"\n    {MAGENTA}SIGINT detected:{NC} Exitting...\n")
    sys.exit(0)


def print_usage() -> None:
    '''
    [PRINT_USAGE] Describes the available command-line flags
    for running the script.
    '''
    usage = f"""
{GREEN}Usage:{NC} python3 train_model.py [1 option]

{YELLOW}Options:{NC}
 -r                         Plot the learning process in real-time.
 -e                         Plot the scaled results of the learning at the end.
 -i                         Plot the input data set used for training and exit.
 -s <learn_rate> <iters>    Set values for the learning rate and iterations.
 -h                         Show this help message.
"""
    print(usage)


def set_flag(opt: str):
    '''
    [SET_FLAG] Sets user-defined options based on command-line arguments.
    '''
    if opt == '-r':
        options['plot_learning'] = True
    elif opt == '-e':
        options['plot_end'] = True
    elif opt == '-i':
        options['plot_input'] = True
    elif opt == '-s':
        options['set_learn_values'] = True
    elif opt == '-h':
        print_usage()
        sys.exit(0)
    else:
        print(f"\n  {RED}Invalid option:{NC} {opt}\n")
        print("  try -h for more information\n")
        sys.exit(1)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        set_flag(sys.argv[1])
        if len(sys.argv) != 2 and not options['set_learn_values']:
            print_usage()
            sys.exit(1)

    if options['set_learn_values']:
        try:
            if len(sys.argv) != 4:
                raise ValueError("input incomplete")
            options['learn_rate'] = float(sys.argv[2])
            options['iters'] = int(sys.argv[3])
            if options['learn_rate'] <= 0 or options['iters'] <= 0:
                raise ValueError("must be positive numbers")
        except ValueError as e:
            print(f"\n  {RED}ValueError:{NC} learn_rate and iterations {e}\n")
            print("  try -h for more information\n")
            sys.exit(1)

    signal.signal(signal.SIGINT, handle_interrupt)
    main()
