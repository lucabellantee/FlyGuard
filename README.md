# FlyGuard - Helicopter Rotor Fault Prediction

## Overview

This project aims to predict potential rotor faults in helicopters using machine learning techniques. The dataset contains various parameters related to the helicopter's rotor system, and the goal is to use these parameters to predict whether the system is faulty or fault-free (binary classification: 0 or 1).

The core of the project involves:
1. **Random Forest Regressor**: To predict the target torque (`trq_target`) based on input features.
2. **Formula Calculation**: Using a known formula to calculate the torque margin (`trq_margin`), a key variable.
3. **Fault Classification**: A Random Forest classifier to determine if the rotor data is faulty or fault-free, based on the calculated `trq_margin` and other input variables.

## Dataset

The dataset consists of the following columns:
- `id`: Unique identifier for each record.
- `trq_measured`: Measured torque (input feature).
- `oat`: Outside air temperature (input feature).
- `mgt`: Main gear temperature (input feature).
- `pa`: Pressure altitude (input feature).
- `ias`: Indicated airspeed (input feature).
- `np`: Rotor speed (input feature).
- `ng`: Gas generator speed (input feature).

These parameters represent critical measurements from the helicopter's rotor system.

## Workflow

1. **Data Preprocessing**: Clean and preprocess the data to handle missing values, outliers, or any other data-related issues.
2. **Feature Engineering**: Derive relevant features such as `trq_target` using the Random Forest regressor.
3. **Torque Margin Calculation**: Apply the known formula to calculate the `trq_margin` based on the predicted `trq_target` and other parameters.
4. **Fault Prediction**: Use a Random Forest classifier to classify the system as faulty (1) or fault-free (0) based on the `trq_margin` and other features.
5. **Model Evaluation**: Evaluate the models using appropriate metrics (e.g., accuracy, precision, recall, etc.).

## Requirements

To run this project, you'll need the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## How to use
1. Clone the repository
   ```bash
   git clone https://github.com/your-username/helicopter-rotor-fault-prediction.git
2. Install the dependencies
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
3. Run the project
   ```bash
   python main.py

