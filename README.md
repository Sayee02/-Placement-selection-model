# Placement-selection-model
# Placement Selection Model

This notebook demonstrates a simple linear regression model to predict package values based on CGPA. It also includes an initial exploration of another dataset with multiple features and a target variable.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model](#model)
5. [Usage](#usage)

## Project Overview
This project contains two main parts:
*   **Multi-feature Linear Regression**: An example of linear regression using a synthetic dataset (`make_regression`) with two features (`feature1`, `feature2`) to predict a `target` variable.
*   **CGPA to Package Prediction**: A simple linear regression model to predict job package (in lakhs per annum) based on a student's CGPA, using the `placement.csv` dataset.

## Installation
This notebook uses common Python libraries for data science. You can install them using pip:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels scipy plotly gradio
```

## Dataset
Two datasets are implicitly used in this notebook:
1.  **Synthetic Dataset**: Generated using `sklearn.datasets.make_regression` with 2 features and 1 target, and some noise. This dataset is used for the multi-feature regression example.
2.  **`placement.csv`**: This CSV file is expected to have two columns: `cgpa` and `package`. It is used to train the simple linear regression model for package prediction.

_Note: The `placement.csv` file should be available in the `/content/` directory or the specified path for the `pd.read_csv` command._

## Model
The core model used in both sections is `LinearRegression` from `sklearn.linear_model`. This model is trained to find a linear relationship between the input features and the target variable.

## Usage

### Multi-feature Regression Example
This section demonstrates basic linear regression concepts:
-   Data generation (`make_regression`)
-   Data splitting (`train_test_split`)
-   Model training (`LinearRegression`)
-   Residual analysis (normality, homoscedasticity)
-   Multicollinearity check (VIF, correlation heatmap)

### CGPA to Package Predictor
1.  **Load Data**: The `placement.csv` file is loaded into a pandas DataFrame.
2.  **Visualize Data**: A scatter plot visualizes the relationship between CGPA and Package.
3.  **Train Model**: A `LinearRegression` model is trained on the CGPA and Package data.
4.  **Prediction**: The trained model can be used to predict packages for new CGPA values.
5.  **Interactive Predictor**: A Gradio interface is created to provide an interactive way to predict packages based on CGPA input.

To run the Gradio app, execute the last code cell of the notebook. It will provide a local URL and a public share link if run in Colab.
