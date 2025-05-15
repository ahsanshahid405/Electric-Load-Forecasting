# Electric Load Forecasting Using Machine Learning

This repository contains an end-to-end implementation of electricity demand forecasting using data mining and machine learning techniques. The project uses the Kaggle dataset of hourly electricity demand and weather measurements for ten major U.S. cities.
# Dataset

We used the dataset "US Top 10 Cities Electricity and Weather Data" from Kaggle, which contains electricity usage and weather data for major US cities.
🔗 Download Dataset from Kaggle
## Dependencies

* Python 3.9+
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost
* tensorflow
* statsmodels
* plotly
* jupyter
  
Install required packages with:  pip install -r requirements.txt

## Introduction

Accurate electricity load forecasting is essential for efficient energy management and grid optimization. Traditional methods often struggle to account for complex interactions between weather, time, and regional demand. This project uses advanced data mining and machine learning techniques to model and predict electricity demand more accurately.

## What Sets Our Solution Apart

* **Dual-Phase Analysis**: We first cluster similar demand-weather patterns, then build predictive models based on these insights.
* **Comprehensive Feature Engineering**: Temporal features (hour, day of week, month, season) and weather data are integrated to model consumption patterns.
* **Ensemble Learning**: Multiple models (bagging, boosting, stacking) are combined to enhance accuracy.
* **Anomaly Detection**: Outliers and erroneous values are detected and handled during preprocessing.

## Addressing Traditional Forecasting Limitations

* **Overreliance on Single Models**: Ensemble methods mitigate weaknesses of individual models.
* **Limited Pattern Recognition**: Clustering uncovers hidden patterns across time and geography.
* **High Sensitivity to Noise**: Preprocessing includes robust anomaly detection and correction.

## Usage

### `01-data-preprocessing-clustering.ipynb`

* Loads and inspects the `cities.csv` dataset.
* Drops columns with excessive missing values (e.g., `precipAccumulation`).
* Handles missing `demand`, `precipType`, `summary`, `icon`, and `ozone` values.
* Applies linear interpolation for continuous variables.
* Extracts temporal features: `hour`, `dayofweek`, `month`, `season`.
* Optionally saves cleaned data to `cleaned_dataset.csv`.
* Applies PCA and t-SNE for visualization.
* Implements K-Means, DBSCAN, and Hierarchical Clustering.
* Uses silhouette scores to evaluate clusters.
* Interprets consumption-weather pattern clusters.

### `03-predictive-modeling.ipynb`

#### Data Preparation

* Loads `cleaned_dataset.csv` and splits into training, validation, and test sets.
* Creates lag features (e.g., `demand` shifted by 24/48/168 hours).
* Generates rolling mean and standard deviation of `demand` (24-hour window).
* Constructs interaction terms: `temperature × hour`, `humidity × hour`.
* Encodes cyclic temporal features (`hour`, `dayofweek`, `month`) using sine/cosine.

#### Model Training

* Trains:

  * Linear Regression (with `SimpleImputer` for missing values)
  * Random Forest
  * XGBoost
  
#### Evaluation

* Metrics used: MAE, RMSE, MAPE
* Performance on validation set:

  * Linear Regression: MAE 4218.31, RMSE 5082.59, MAPE 180.84
  * Random Forest: MAE 301.22, RMSE 679.07, MAPE 7.73
  * XGBoost: MAE 851.99, RMSE 1166.21, MAPE 36.32

#### Optimization

* Applies `GridSearchCV` to optimize Random Forest.
* Best hyperparameters: `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=2`, `min_samples_split=2`, `n_estimators=100`
* Optimized Random Forest: MAE 591.91, RMSE 808.85, MAPE 24.99

#### Ensemble Learning

* Stacking ensemble:

  * Base models: Random Forest, XGBoost
  * Meta-model: Linear Regression
* Test set performance:

  * MAE 599.20, RMSE 871.62, MAPE 5.82

#### Baseline Comparison

* Naive baseline (same hour previous day): MAE 870.51, RMSE 1265.50, MAPE 8.20
* Ensemble improvements: 31.17% (MAE), 31.12% (RMSE), 29.04% (MAPE)

#### Model Saving

* Models saved as `random_forest_model.pkl`, `xgboost_model.pkl`, `meta_model.pkl` using `joblib`

