import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold as skfold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import BernoulliNB as bernoulli
from sklearn.linear_model import LogisticRegression as logistic_regr
from sklearn.ensemble import RandomForestClassifier as random_forest
from sklearn.svm import SVC as svc
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.naive_bayes import MultinomialNB as naiveB
from sklearn.ensemble import VotingClassifier as voting_
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import log_loss, make_scorer, precision_score, accuracy_score, f1_score, confusion_matrix, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")

train_data = pd.read_csv('/content/train.csv')
test_data= pd.read_csv('/content/test.csv')

# Check for missing values
missing_values = train_data.isna().sum().sum()
print(f'There are {missing_values} missing feature values.\n')

# Check for duplicate rows
duplicate_rows = len(train_data) - len(train_data.drop_duplicates())
print(f'There are {duplicate_rows} duplicate rows.\n')

# Print unique target values
unique_targets = train_data['prognosis'].nunique()
print(f'There are {unique_targets} unique prognoses (Labels).\n')

# Get list of unique features and print quantity
features = train_data.columns.difference(['id', 'prognosis'])
num_unique_features = len(features)
print(f'There are {num_unique_features} unique symptoms (Features).\n')

# Get length of train_data and test datasets
print(f'train_data dataset length: {len(train_data)}')
print(f'test_data dataset length: {len(test_data)}\n\n')

# Count the occurrences of each unique prognosis
prognosis_counts = train_data['prognosis'].value_counts()

# Plot a pie chart
plt.figure(figsize=(8, 8))
plt.pie(prognosis_counts, labels=prognosis_counts.index, startangle=140, counterclock=False,autopct=lambda p:f'{int(p*sum(prognosis_counts)/100)}')
plt.title('Distribution of Prognoses')
plt.show()

# Calculate the correlation matrix for numeric columns
numeric_columns = train_data.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()
# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Encoding labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(train_data['prognosis'])

# Splitting the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(train_data[features],
                                                      encoded_labels,
                                                      train_size=0.80,
                                                      shuffle=True,
                                                      random_state=2,
                                                      stratify=train_data[['prognosis']])
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
# Create a PCA instance with 15 components and fit_transform
k=35
pca= PCA(n_components=k)
X_transformed_pca = pca.fit_transform(X_train)

# Display a sample of the transformed data with explained variance of the principal components
print(X_transformed_pca[:5])  # Display the first 5 rows of transformed data
explained_variance_ratio= pca.explained_variance_ratio_
for i in range(len(explained_variance_ratio)):
    print(f'PC_{i} variance explained ratio: {np.round(explained_variance_ratio[i], decimals=3)}')

# Plotting the transformed data along the principal component axes
fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(data=pd.concat([pd.DataFrame(X_transformed_pca, columns=[f'pca{i}' for i in range(k)]),
                                train_data[['prognosis']]], axis=1), x='pca0', y='pca1', hue='prognosis', palette='tab10').set_title('PCA Trial', fontsize=20)
fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(data=pd.concat([pd.DataFrame(X_transformed_pca, columns=[f'pca{i}' for i in range(k)]),
                                train_data[['prognosis']]], axis=1), x='pca1', y='pca2', hue='prognosis', palette='tab10').set_title('PCA Trial', fontsize=20)
plt.show()

# Rank the features based on their contributions to PCA components
feature_ranking = np.argsort(explained_variance_ratio)[::-1]

# Select the top 35 features based on explained variance ratio
top_35_features = X_train.columns[feature_ranking[:k]]
# Select the top 35 features based on feature_ranking
top_35_features = X_train.columns[feature_ranking[:k]]

# Create X_train_pca with only the top 35 features
X_train_pca = X_train[top_35_features]

# Create X_valid_pca with only the top 35 features
X_valid_pca = X_valid[top_35_features]

# Create test_data with only the top 35 features
test_data=test_data[top_35_features]