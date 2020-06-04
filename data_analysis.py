import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
import statsmodels.api as sm
from sklearn.linear_model import LassoCV

# Read dataset
data_frame = pd.read_csv('dataset/insurance.csv')

# Transform labels to category for plot correlation matrix
plot_data = data_frame.copy()
for x in ['sex', 'smoker', 'region']:
    plot_data[x] = plot_data[x].astype('category')
cat_columns = plot_data.select_dtypes(['category']).columns
plot_data[cat_columns] = plot_data[cat_columns].apply(lambda x: x.cat.codes)

# Get and plot correlation matrix
corrMatrix = plot_data.corr()
plt.figure(figsize=(10, 8))
sn.heatmap(corrMatrix, annot=True, fmt='g', )
plt.show()

# Checking dataset for missing data
num_missing = 0
for col in plot_data.columns:
    missing = plot_data[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        print('created missing indicator for: {}'.format(col))
        plot_data['{}_ismissing'.format(col)] = missing

# Plot histogram with missing data
ismissing_cols = [col for col in plot_data.columns if 'ismissing' in col]
plot_data['num_missing'] = plot_data[ismissing_cols].sum(axis=1)
plot_data['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')
plt.show()

# Print percentage of missing data by feature
for col in plot_data.columns:
    pct_missing = np.mean(plot_data[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing * 100)))
print('-' * 8)
# Plot boxplot
plot_data.boxplot(column=['bmi'])
plt.show()
sn.boxplot(x=['charges'], data=plot_data)
plt.show()

# Search for uninformative features
num_rows = len(plot_data.index)
low_information_cols = []  #

for col in plot_data.columns:
    cnts = plot_data[col].value_counts(dropna=False)
    top_pct = (cnts / num_rows).iloc[0]

    if top_pct > 0.95:
        low_information_cols.append(col)
        print('{0}: {1:.5f}%'.format(col, top_pct * 100))
        print(cnts)
        print()

# Search important features with ExtraTrees and Backward Elimination

array = plot_data.values
X = array[:, 0:5]
Y = array[:, 6]

model = ExtraTreesRegressor()
model.fit(X, Y)
print(plot_data.columns)
print(model.feature_importances_)

# Backward Elimination
X = plot_data.drop('charges', 1
                   )
Y = plot_data['charges']
cols = list(X.columns)
pmax = 1
while (len(cols) > 0):
    p = []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(Y, X_1).fit()
    p = pd.Series(model.pvalues.values[1:], index=cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if (pmax > 0.1):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

# Transform labels to category with OneHotEncoder
data_frame_category = data_frame.copy()
data_frame_category = data_frame_category.select_dtypes(include=['object'])
data_frame_encoded = data_frame_category.copy()

data_label_encoded = pd.get_dummies(data_frame_encoded, columns=['sex', 'region', 'smoker'])
data_frame = data_frame.drop(['sex', 'region', 'smoker'], axis=1)

# Result encoded dataset
data_encoded = pd.concat([data_frame, data_label_encoded], axis=1)

# Search important features with ExtraTrees and Backward Elimination for OneHotEncoder

Y = data_encoded['charges']
X = data_encoded.drop('charges', 1)
X = X.values

model = ExtraTreesRegressor()
model.fit(X, Y)
print(data_encoded.columns)
print(model.feature_importances_)

# Correlation matrix for OneHotEncoder
corrMatrix = data_encoded.corr()
plt.figure(figsize=(10, 8))
sn.heatmap(corrMatrix, annot=True, fmt='g', )
plt.show()

# Backward Elimination for OneHotEncoder

Y = data_encoded['charges']
X = data_encoded.drop('charges', 1)
cols = list(X.columns)
pmax = 1
while (len(cols) > 0):
    p = []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(Y, X_1).fit()
    p = pd.Series(model.pvalues.values[1:], index=cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if (pmax > 0.1):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

# Embedded Method

reg = LassoCV()
reg.fit(X, Y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" % reg.score(X, Y))
coef = pd.Series(reg.coef_, index=X.columns)

print(
    "Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()