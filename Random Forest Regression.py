import numpy as np  # For numerical computations
import pandas as pd  # For handling datasets
import matplotlib.pyplot as plt  # For visualizations

data = pd.read_csv(r"D:\FSDS Material\Dataset\Non Linear emp_sal.csv")

# Extract independent variable (x) and dependent variable (y)
x = data.iloc[:, 1:2].values 
y = data.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=30,
                                     criterion='absolute_error',
                                     min_samples_split=5,
                                     random_state=0)
rf_regressor.fit(x,y)

rf_reg_pred = rf_regressor.predict([[6.5]])
print(rf_reg_pred)
