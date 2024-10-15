from train_model import load_data
import sys
import os
import numpy as np
from typing import Tuple
from yaml import load, FullLoader

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
NC = "\033[0m"


def rescale_value(value: float, dataset: np.ndarray, mode: int) -> float:
    '''
    [RESCALE_VALUE] Rescales (mode 1) or descales (mode 2) a value based
    on the dataset using min-max normalization.
    '''
    min_val = np.min(dataset)
    max_val = np.max(dataset)
    if mode == 1:
        return ((value - min_val) / (max_val - min_val))
    elif mode == 2:
        return ((value * (max_val - min_val)) + min_val)


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    '''
    [ESTIMATE_PRICE] Estimates the price of a car based on its mileage
    using a linear regression model.
    '''
    mileage_set, price_set = load_data("data.csv")
    mileage = rescale_value(mileage, mileage_set, 1)
    price = theta0 + (theta1 * mileage)
    price = rescale_value(price, price_set, 2)
    return price


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


def main():
    '''
    [MAIN] Handles user input for mileage and estimates the price of
    a car using a linear regression model.
    '''
    try:
        mileage_val = float(input("\n  Enter mileage: "))
        if mileage_val < 0:
            raise ValueError("mileage has to be a positive number")

    except ValueError as e:
        print(f"\n  {RED}ValueError:{NC} {e}\n")
        sys.exit(1)
    except EOFError:
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"\n  {RED}An unexpected error occured:{NC} {e}\n")
        sys.exit(1)

    if not os.path.isfile('thetas.yaml'):
        print(f"\n  {YELLOW}[WARNING]{NC} The model has not been trained yet.")
        print(f"\n    {GREEN}Estimated price:{NC} 0\n")
        return
    else:
        theta0, theta1 = load_thetas("thetas.yaml")

    price = estimate_price(mileage_val, theta0, theta1)
    if price < 0:
        print(f"\n    {RED}Estimated price:{NC} negative value, consider "
              "lowering the mileage\n")
        return
    print(f"\n    {GREEN}Estimated price:{NC} {price:.1f}\n")


if __name__ == "__main__":
    main()
