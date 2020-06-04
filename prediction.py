import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lime
import lime.lime_tabular


def graphics_plot(y_test, y_pred, title):
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    print(df)

    # Plot histogram with actual and predicted values
    df1 = df.head(25)
    df1.plot(kind='bar', figsize=(16, 10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title(title)
    plt.show()

    # Print performance of the algorithm
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Read the data
data_frame = pd.read_csv('dataset/insurance.csv')

###### PREPROCESSING DATA ########

# Transform labels to category with OneHotEncoder
data_frame_category = data_frame.copy()
data_frame_category = data_frame_category.select_dtypes(include=['object'])
data_frame_encoded = data_frame_category.copy()

data_label_encoded = pd.get_dummies(data_frame_encoded, columns=['sex', 'region', 'smoker'])
data_frame = data_frame.drop(['sex', 'region', 'smoker'], axis=1)

# Result encoded dataset
data_encoded = pd.concat([data_frame, data_label_encoded], axis=1)

# Standardisation
scaler = RobustScaler()

Y = data_encoded['charges']
data_encoded = data_encoded.drop(columns=['charges','sex_female', 'sex_male', 'region_northeast',
      'region_northwest', 'region_southeast', 'region_southwest','children'], axis=1)
print(data_encoded.columns)
y = Y.values.reshape(-1, 1)
X_scal = data_encoded[['age', 'bmi']]
X_not_scal = data_encoded[['smoker_no',
      'smoker_yes']]
scaler.fit_transform(X_scal, y)
X = np.concatenate((X_scal, X_not_scal), axis=1)
X = data_encoded
# Divide the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)



# Linear regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)

# Print actual and predicted and plot it
graphics_plot(y_test, y_pred, 'Linear Regression with 3 columns')

# Decision Tree
decision_regressor = DecisionTreeRegressor(random_state=1,criterion='mae')
decision_regressor.fit(X_train, y_train)
y_pred = decision_regressor.predict(X_test)

# Print actual and predicted and plot it
graphics_plot(y_test,y_pred,'DecisionTree Regressor with 3 columns')


xgb_regressor = xgb.XGBRegressor(colsample_bytree=0.4,
                                      gamma=0,
                                      learning_rate=0.07,
                                      max_depth=2,
                                      min_child_weight=1.5,
                                      n_estimators=1000,
                                      reg_alpha=0.75,
                                      reg_lambda=0.45,
                                      subsample=0.6,
                                      seed=42)

xgb_regressor.fit(X_train, y_train)
y_pred = xgb_regressor.predict(X_test)
graphics_plot(y_test, y_pred, 'XGBoost with 3 columns')
