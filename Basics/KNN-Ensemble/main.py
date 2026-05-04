import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visualizations
sns.set(style="whitegrid")

df = pd.read_csv('dataset.csv')
# Check the basic structure of the dataset
print("Dataset Info:")
df.info()  # Check data types and for missing values

print("\nSummary Statistics:")
print(df.describe())  # Summary statistics of numeric features

# Check for any missing values in the dataset
print("\nMissing Values:")
print(df.isnull().sum())  # Find if any column has missing values

# Explore the target variable (binary classification)
print("\nTarget Variable Distribution:")
print(df['Attrition'].value_counts())  # Distribution of the target classes

# Measure and print the number of unique values for each column.
# Check if there are any columns with less than 2 unique values. If so, remove them.

unique_counts= {}
for col in df:
    unique_counts[col]= df[col].nunique()
print(unique_counts)

print("colmuns with less than 2 unique values: ")
for key, value in unique_counts.items():
    if value<2:
        print(key)
        df.drop(key, axis= 1, inplace= True)

# Label encode all categorical columns

encoded_df= df.copy()
for col in encoded_df.columns:
    if encoded_df[col].dtype == 'object':
        encoded_df[col]=LabelEncoder().fit_transform(encoded_df[col])
encoded_df.head(5)

# Split into features and target variable
X = encoded_df.drop(columns=['Attrition'])
y = encoded_df['Attrition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)