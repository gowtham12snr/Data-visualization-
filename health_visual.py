# -*- coding: utf-8 -*-
"""Health_Visual.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19j-aKrKkbKqaUhJNb-Rf_EizL9yBjgHL

# Importing the Libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

"""# Load the dataset (assuming the CSV file is named 'healthcare_dataset.csv')"""

healthcare_data = pd.read_csv('healthcare_dataset.csv')

"""# Display an introductory sneak peek of the data"""

healthcare_data.head()

"""## Display the shape of the dataset"""

print(f'The dataset contains {healthcare_data.shape[0]} rows and {healthcare_data.shape[1]} columns.')

"""## Get a summary of the dataset"""

healthcare_data.info()

"""## Check for missing values in the dataset"""

missing_values = healthcare_data.isnull().sum()
print(f'\nMissing values in each column:\n{missing_values}')

"""## Standardize the 'Name' column by converting it to lowercase"""

healthcare_data['Name'] = healthcare_data['Name'].str.lower()

# Display the updated DataFrame
healthcare_data.head()

# Convert date columns to datetime format
healthcare_data['Date of Admission'] = pd.to_datetime(healthcare_data['Date of Admission'])
healthcare_data['Discharge Date'] = pd.to_datetime(healthcare_data['Discharge Date'])

# Provide a statistical summary of numerical columns
display(healthcare_data.describe())

"""# Provide a summary of categorical columns"""

display(healthcare_data.describe(include="object").T)

"""# Display the distribution of categorical columns"""

print(healthcare_data['Gender'].value_counts())

print(healthcare_data['Blood Type'].value_counts())

print(healthcare_data['Admission Type'].value_counts())

print(healthcare_data['Insurance Provider'].value_counts())

print(healthcare_data['Doctor'].value_counts().sum())
print(healthcare_data['Test Results'].value_counts())

"""# Visualizations"""

# Histogram for age distribution
age_distribution_fig = px.histogram(healthcare_data, x='Age', title='Distribution of Age among Patients', nbins=40, color_discrete_sequence=['#87CEEB'])
age_distribution_fig.update_layout(xaxis_title='Age', yaxis_title='Count', title_font_size=24)
age_distribution_fig.show()

# List of categorical columns for visualizations
categorical_columns = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Insurance Provider', 'Medication', 'Test Results']

# Set a new color palette
new_palette = px.colors.qualitative.Vivid

# Create visualizations for each categorical column
for col in categorical_columns:
    fig = go.Figure()
    for idx, (category, count) in enumerate(healthcare_data[col].value_counts().items()):
        fig.add_trace(go.Bar(x=[col], y=[count], name=category, marker_color=new_palette[idx]))
    fig.update_layout(title=f'Distribution of {col}', xaxis_title=col, yaxis_title='Count', title_font_size=24)
    fig.show()

# Calculate the average age by medical condition
average_age_by_condition = healthcare_data.groupby('Medical Condition')['Age'].mean().reset_index()

# Visualize average age by medical condition
age_condition_fig = px.bar(average_age_by_condition, x='Medical Condition', y='Age', color='Medical Condition',
                           title='Average Age of Patients by Medical Condition',
                           labels={'Age': 'Average Age', 'Medical Condition': 'Condition'},
                           color_discrete_sequence=new_palette)
age_condition_fig.update_layout(title_font_size=24)
age_condition_fig.show()

# Grouping by Medical Condition and Medication to count occurrences
medication_distribution = healthcare_data.groupby(['Medical Condition', 'Medication']).size().reset_index(name='Count')

# Visualizing medication distribution by medical condition
med_condition_fig = px.bar(medication_distribution, x='Medical Condition', y='Count', color='Medication', barmode='group',
                           title='Distribution of Medication by Medical Condition',
                           labels={'Count': 'Number of Patients', 'Medical Condition': 'Condition', 'Medication': 'Medication'},
                           color_discrete_sequence=new_palette)
med_condition_fig.update_layout(title_font_size=24)
med_condition_fig.show()

# Analyze patient distribution by gender and medical condition
gender_condition_distribution = healthcare_data.groupby(['Medical Condition', 'Gender']).size().reset_index(name='Count')

gender_condition_fig = px.bar(gender_condition_distribution,
                              x='Medical Condition',
                              y='Count',
                              color='Gender',
                              barmode='group',
                              title='Patient Distribution by Gender and Medical Condition',
                              labels={'Count': 'Number of Patients', 'Medical Condition': 'Condition', 'Gender': 'Gender'},
                              color_discrete_sequence=px.colors.qualitative.Set1)

gender_condition_fig.update_layout(title_font_size=24)
gender_condition_fig.show()

# Analyze patient count by blood type and medical condition
blood_condition_distribution = healthcare_data.groupby(['Blood Type', 'Medical Condition']).size().reset_index(name='Count')

# Visualization for patient count by blood type and medical condition
blood_condition_fig = px.bar(blood_condition_distribution,
                             x='Blood Type',
                             y='Count',
                             color='Medical Condition',
                             barmode='group',
                             title='Patient Distribution by Blood Type and Medical Condition',
                             labels={'Count': 'Number of Patients', 'Blood Type': 'Blood Type', 'Medical Condition': 'Condition'},
                             color_discrete_sequence=px.colors.qualitative.Vivid)

blood_condition_fig.update_layout(title_font_size=24)
blood_condition_fig.show()

# Analyze patient count by blood type and gender
blood_gender_distribution = healthcare_data.groupby(['Blood Type', 'Gender']).size().reset_index(name='Count')

# Visualization for patient count by blood type and gender
blood_gender_fig = px.bar(blood_gender_distribution, x='Blood Type', y='Count', color='Gender', barmode='group',
                          title='Patient Distribution by Blood Type and Gender',
                          labels={'Count': 'Number of Patients', 'Blood Type': 'Blood Type', 'Gender': 'Gender'},
                          color_discrete_sequence=new_palette)
blood_gender_fig.update_layout(title_font_size=24)
blood_gender_fig.show()

# Analyze patient count by admission type and gender
admission_gender_distribution = healthcare_data.groupby(['Admission Type', 'Gender']).size().reset_index(name='Count')

# Visualization for patient count by admission type and gender
admission_gender_fig = px.bar(admission_gender_distribution, x='Admission Type', y='Count', color='Gender', barmode='group',
                               title='Patient Distribution by Admission Type and Gender',
                               labels={'Count': 'Number of Patients', 'Admission Type': 'Admission Type', 'Gender': 'Gender'},
                               color_discrete_sequence=new_palette)
admission_gender_fig.update_layout(title_font_size=24)
admission_gender_fig.show()

# Analyze patient count by admission type and medical condition
admission_condition_distribution = healthcare_data.groupby(['Admission Type', 'Medical Condition']).size().reset_index(name='Count')

# Visualization for patient count by admission type and medical condition
admission_condition_fig = px.bar(admission_condition_distribution, x='Admission Type', y='Count', color='Medical Condition', barmode='group',
                                 title='Patient Distribution by Admission Type and Medical Condition',
                                 labels={'Count': 'Number of Patients', 'Admission Type': 'Admission Type', 'Medical Condition': 'Condition'},
                                 color_discrete_sequence=new_palette)
admission_condition_fig.update_layout(title_font_size=24)
admission_condition_fig.show()

# Analyze test results by admission type
test_admission_distribution = healthcare_data.groupby(['Test Results', 'Admission Type']).size().reset_index(name='Count')

# Visualization for test results distribution by admission type
test_admission_fig = px.bar(test_admission_distribution, x='Test Results', y='Count', color='Admission Type', barmode='group',
                            title='Distribution of Test Results by Admission Type',
                            labels={'Count': 'Number of Tests', 'Test Results': 'Test Results', 'Admission Type': 'Admission Type'},
                            color_discrete_sequence=new_palette)
test_admission_fig.update_layout(title_font_size=24)
test_admission_fig.show()

# Analyze medication distribution by gender
medication_gender_distribution = healthcare_data.groupby(['Medication', 'Gender']).size().reset_index(name='Count')

# Visualization for medication distribution by gender
med_gender_fig = px.bar(medication_gender_distribution, x='Medication', y='Count', color='Gender', barmode='group',
                        title='Medication Distribution by Gender',
                        labels={'Count': 'Number of Prescriptions', 'Medication': 'Medication', 'Gender': 'Gender'},
                        color_discrete_sequence=new_palette)
med_gender_fig.update_layout(title_font_size=24)
med_gender_fig.show()

"""# Data Pre processing

## Select relevant features and target variable
"""

features = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Insurance Provider', 'Medication', 'Test Results']
target = 'Medical Condition'

"""## Encode categorical variables"""

label_encoders = {}
for col in features:
    if healthcare_data[col].dtype == 'object':
        le = LabelEncoder()
        healthcare_data[col] = le.fit_transform(healthcare_data[col])
        label_encoders[col] = le

# Define features (X) and target (y)
X = healthcare_data[features]
y = healthcare_data[target]

"""## Split the data into training and testing sets"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

"""## Standardize the feature set"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""# Logistic Regression model"""

# Initialize and train Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Predict and evaluate Logistic Regression model
y_pred_log_reg = log_reg.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

print("Logistic Regression Accuracy:", accuracy_log_reg)
print(f'Classification Report:\n{classification_report(y_test, y_pred_log_reg)}')

# Generate confusion matrix
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Logistic Regression', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.show()

"""# Random Forest model"""

# Initialize and train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate Random Forest model
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {accuracy_rf}")
print(f'Classification Report:\n{classification_report(y_test, y_pred_rf)}')

# Generate confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='inferno', cbar=False)
plt.title('Confusion Matrix for Random Forest', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.show()

"""# Support Vector Machine model"""

# Initialize and train Support Vector Machine model
svm = SVC(random_state=42)
svm.fit(X_train, y_train)

# Predict and evaluate SVM model
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print("Accuracy of SVM model:", accuracy_svm)
print(f'Classification Report:\n{classification_report(y_test, y_pred_svm)}')

# Generate confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title('Confusion Matrix for Support Vector Machine (SVM)', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.show()

"""# Plot the accuracy comparison"""

# Store the accuracy of each model in a dictionary
model_accuracies = {
    'Logistic Regression': accuracy_log_reg,
    'Random Forest': accuracy_rf,
    'Support Vector Machine (SVM)': accuracy_svm
}

# Convert the dictionary into a DataFrame for easier visualization
accuracy_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])


plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=accuracy_df, palette='viridis')
plt.title('Comparison of Model Accuracies', fontsize=18)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1)
plt.show()