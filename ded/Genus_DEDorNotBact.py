#%% Preprocessing
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import tqdm
from timeit import default_timer as timer

from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#%%
os.chdir('c:/users/fre_f/pythonprojects/bacteriological/')

# %%
df = pd.read_csv('./data/Genus_DEDorNot.csv')
# %%
df.head()
# %%
df.iloc[:,2:600]
# %%
print(df.iloc[:,1])
# %%
X = df.iloc[:,2:600]
y = df.iloc[:,1]
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# %%
def xgbmetrics():
    y_pred = clfxgb.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cf = confusion_matrix(y_test, y_pred)
    target_names = ['1', '0']
    print(f"Balanced accuracy: {bal_acc}")
    print(f"MCC: {mcc}")
    print(f"F1: {f1}")
    print(f"Confusion matrix: \n{cf}")
    print(f"Classification report: \n{classification_report(y_test, y_pred, target_names=target_names)}")
# %%
def lgbmetrics():
    y_pred = clflgbm.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cf = confusion_matrix(y_test, y_pred)
    target_names = ['1', '0']
    print(f"Balanced accuracy: {bal_acc}")
    print(f"MCC: {mcc}")
    print(f"F1: {f1}")
    print(f"Confusion matrix: \n{cf}")
    print(f"Classification report: \n{classification_report(y_test, y_pred, target_names=target_names)}")

# %% Train and predict
    
clfxgb = XGBClassifier().fit(X_train, y_train)
clfxgb.predict(X_test)

xgbmetrics()

# %%
import lightgbm as lgb
from lightgbm import LGBMClassifier
clflgbm = LGBMClassifier().fit(X_train, y_train)
clflgbm.predict(X_test)

lgbmetrics()
# %%
from xgboost import plot_importance
fig, ax = plt.subplots()
plot_importance(clfxgb, max_num_features = 20, ax=ax)
plt.title('Feature importance')

plt.show()

#%%
# Create a SHAP explainer for the XGBoost model
explainerxgb = shap.Explainer(clfxgb, X_test)

# Compute SHAP values for the dataset
shap_valuesxgb = explainerxgb(X_test)

# Plot the SHAP summary plot

shap.plots.bar(shap_valuesxgb)

#%%
# Plot the SHAP dependence plot for a specific feature
# Replace 'feature_name' with the name of the feature you want to visualize
feature_name = 'Acinetobacter'  
shap.plots.scatter(shap_valuesxgb[:, feature_name], color=shap_valuesxgb)

#%%
# Plot the SHAP decision plot for a specific instance
# Replace 'instance_index' with the index of the instance you want to visualize
instance_index = 0  # Example instance index
shap.plots.waterfall(shap_valuesxgb[instance_index])


# %%
lgb.plot_importance(clflgbm)
# %%
# Create a SHAP explainer for the LightGBM model

explainer = shap.Explainer(clflgbm, X_test)

# Compute SHAP values for the dataset

shap_values = explainer(X_test)

# Plot the SHAP summary plot

shap.plots.bar(shap_values)

#%%
# Plot the SHAP dependence plot for a specific feature
# Replace 'feature_name' with the name of the feature you want to visualize
feature_name = 'Anaerococcus'  # Example feature name
shap.plots.scatter(shap_values[:, feature_name], color=shap_values)

#%%
# Plot the SHAP decision plot for a specific instance
# Replace 'instance_index' with the index of the instance you want to visualize
instance_index = 0  # Example instance index
shap.plots.waterfall(shap_values[instance_index])