# ft_linear_regression

This project implements a basic machine learning model, focusing on linear regression to predict the price of a car based on its mileage. It uses a linear model and gradient descent to train the model.

## Objective

The goal is to implement a simple linear regression model that predicts car prices based on mileage data. The project consists of three main programs:

1. **Prediction Program**: Estimates the price of the car based on a given mileage value.
2. **Training Program**: Trains the model using a dataset and updates the model's parameters (`theta0` and `theta1`) using gradient descent.
3. **Precision Program**: Calculates the precision of the algorithm using value statistics, MAE, MSE, and RMSE.

## Usage

1. **Prediction Program**:
   - prompts the user for a mileage value
   - returns the estimated price using the hypothesis formula

2. **Training Program**:
   - reads the dataset from the `data.csv` file
   - trains the model by performing linear regression using gradient descent
   - exports the parameters `theta0` and `theta1` into a `thetas.yaml` file

3. **Precision Program**:
   - reads the dataset `data.csv` and the model parameters from `thetas.yaml` file
   - computes value statistics and error metrics


## Linear regression

The linear regression model is based on the following hypothesis formula, which corresponds to the equation of a straight line:

$$ \text{estimatePrice}(mileage) = \theta_0 + (\theta_1 \times mileage) $$

The gradient descent algorithm calculates corrections for both `theta` parameters as follows:

$$ corrGrad_0 = \text{learningRate} \times \frac{1}{\text{members}} \sum_{i=0}^{\text{members}-1} \left( \text{estimatePrice}(mileage[i]) - price[i] \right) $$

$$ corrGrad_1 = \text{learningRate} \times \frac{1}{\text{members}} \sum_{i=0}^{\text{members}-1} \left( \text{estimatePrice}(mileage[i]) - price[i] \right) \times mileage[i] $$

After each iteration of the training cycle, the parameters `theta0` and `theta1` are updated:

$$ \theta_x = \theta_x - corr\_grad_x $$

Due to the differing scales of the data in the dataset, both the mileage (feature) and price (target) values are rescaled using min-max normalization:

$$ x' = \frac{x - min(x)}{max(x) - min(x)} $$


## Precision
The precision of the model is measured using various error metrics, which evaluate how well the predictions match the actual values.

1. **Mean Absolute Error (MAE):** Measures the average magnitude of errors in a set of predictions, without considering their direction

$$ \text{MAE} = \frac{1}{m} \sum_{i=1}^{m} \left| \text{predicted}[i] - \text{real}[i] \right| $$

2. **Mean Squared Error (MSE):** Penalizes larger errors more heavily by squaring them before averaging

$$ \text{MSE} = \frac{1}{m} \sum_{i=1}^{m} \left( \text{predicted}[i] - \text{real}[i] \right)^2 $$

3. **Root Mean Squared Error (RMSE):** Provides error values in the same unit as the target variable

$$ \text{RMSE} = \sqrt{ \frac{1}{m} \sum_{i=1}^{m} \left( \text{predicted}[i] - \text{real}[i] \right)^2 } $$


## Installation

1. Clone the repository:
``` bash
git clone https://github.com/farkasf/ft_linear_regression.git
```

2. Activate the workdir creation script:
``` bash
bash create_workdir.sh && cd workdir
```

3. Use python3 to execute the scripts.
``` bash
python3 train_model.py -h
```
