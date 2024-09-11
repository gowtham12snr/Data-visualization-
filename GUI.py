import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tkinter as tk
from tkinter import Scale, Button
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the main window
root = Tk()
root.title("Health Insurance Data Analysis")
root.geometry("1200x700")

# Function to load the dataset
def load_data():
    global healthcare_data
    file_path = filedialog.askopenfilename()
    if file_path:
        healthcare_data = pd.read_csv(file_path)
        messagebox.showinfo("Success", "Dataset loaded successfully!")


# Function for displaying the first few rows of the dataset
def show_head():
    global healthcare_data
    if healthcare_data is not None:
        print(healthcare_data.head())

# Function for displaying the dataset's shape
def show_shape():
    global healthcare_data
    if healthcare_data is not None:
        print(f'The dataset contains {healthcare_data.shape[0]} rows and {healthcare_data.shape[1]} columns.')

# Function for checking missing values
def check_missing_values():
    global healthcare_data
    if healthcare_data is not None:
        missing_values = healthcare_data.isnull().sum()
        print(f'\nMissing values in each column:\n{missing_values}')

# Function for standardizing the 'Name' column
def standardize_name():
    global healthcare_data
    if healthcare_data is not None:
        healthcare_data['Name'] = healthcare_data['Name'].str.lower()
        print(healthcare_data.head())

# Function for converting date columns to datetime format
def convert_dates():
    global healthcare_data
    if healthcare_data is not None:
        healthcare_data['Date of Admission'] = pd.to_datetime(healthcare_data['Date of Admission'])
        healthcare_data['Discharge Date'] = pd.to_datetime(healthcare_data['Discharge Date'])
        print("Dates converted to datetime format.")

# Function to perform EDA visualizations
def perform_eda():
    global healthcare_data
    if healthcare_data is not None:
        # Histogram for age distribution
        age_distribution_fig = px.histogram(healthcare_data, x='Age', title='Distribution of Age among Patients', nbins=40, color_discrete_sequence=['#87CEEB'])
        age_distribution_fig.update_layout(xaxis_title='Age', yaxis_title='Count', title_font_size=24)
        age_distribution_fig.show()

        # List of categorical columns for visualizations
        categorical_columns = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Insurance Provider', 'Medication', 'Test Results']
        new_palette = px.colors.qualitative.Vivid

        # Create visualizations for each categorical column
        for col in categorical_columns:
            fig = go.Figure()
            for idx, (category, count) in enumerate(healthcare_data[col].value_counts().items()):
                fig.add_trace(go.Bar(x=[col], y=[count], name=category, marker_color=new_palette[idx]))
            fig.update_layout(title=f'Distribution of {col}', xaxis_title=col, yaxis_title='Count', title_font_size=24)
            fig.show()

        # Average age by medical condition
        average_age_by_condition = healthcare_data.groupby('Medical Condition')['Age'].mean().reset_index()
        age_condition_fig = px.bar(average_age_by_condition, x='Medical Condition', y='Age', color='Medical Condition',
                                   title='Average Age of Patients by Medical Condition',
                                   labels={'Age': 'Average Age', 'Medical Condition': 'Condition'},
                                   color_discrete_sequence=new_palette)
        age_condition_fig.update_layout(title_font_size=24)
        age_condition_fig.show()

# Define your color palettes
new_palette = px.colors.qualitative.Plotly

# Function to plot the selected visualization
def plot_visualization():
    # Get the selected visualization option from the slider
    visualization_option = vis_slider.get()

    if visualization_option == 1:
        # Average Age by Medical Condition
        average_age_by_condition = healthcare_data.groupby('Medical Condition')['Age'].mean().reset_index()
        age_condition_fig = px.bar(average_age_by_condition, x='Medical Condition', y='Age', color='Medical Condition',
                                   title='Average Age of Patients by Medical Condition',
                                   labels={'Age': 'Average Age', 'Medical Condition': 'Condition'},
                                   color_discrete_sequence=new_palette)
        age_condition_fig.update_layout(title_font_size=24)
        age_condition_fig.show()

    elif visualization_option == 2:
        # Distribution of Medication by Medical Condition
        medication_distribution = healthcare_data.groupby(['Medical Condition', 'Medication']).size().reset_index(name='Count')
        med_condition_fig = px.bar(medication_distribution, x='Medical Condition', y='Count', color='Medication', barmode='group',
                                   title='Distribution of Medication by Medical Condition',
                                   labels={'Count': 'Number of Patients', 'Medical Condition': 'Condition', 'Medication': 'Medication'},
                                   color_discrete_sequence=new_palette)
        med_condition_fig.update_layout(title_font_size=24)
        med_condition_fig.show()

    elif visualization_option == 3:
        # Patient Distribution by Gender and Medical Condition
        gender_condition_distribution = healthcare_data.groupby(['Medical Condition', 'Gender']).size().reset_index(name='Count')
        gender_condition_fig = px.bar(gender_condition_distribution, x='Medical Condition', y='Count', color='Gender', barmode='group',
                                      title='Patient Distribution by Gender and Medical Condition',
                                      labels={'Count': 'Number of Patients', 'Medical Condition': 'Condition', 'Gender': 'Gender'},
                                      color_discrete_sequence=px.colors.qualitative.Set1)
        gender_condition_fig.update_layout(title_font_size=24)
        gender_condition_fig.show()

    elif visualization_option == 4:
        # Patient Distribution by Blood Type and Medical Condition
        blood_condition_distribution = healthcare_data.groupby(['Blood Type', 'Medical Condition']).size().reset_index(name='Count')
        blood_condition_fig = px.bar(blood_condition_distribution, x='Blood Type', y='Count', color='Medical Condition', barmode='group',
                                     title='Patient Distribution by Blood Type and Medical Condition',
                                     labels={'Count': 'Number of Patients', 'Blood Type': 'Blood Type', 'Medical Condition': 'Condition'},
                                     color_discrete_sequence=px.colors.qualitative.Vivid)
        blood_condition_fig.update_layout(title_font_size=24)
        blood_condition_fig.show()

    elif visualization_option == 5:
        # Patient Distribution by Blood Type and Gender
        blood_gender_distribution = healthcare_data.groupby(['Blood Type', 'Gender']).size().reset_index(name='Count')
        blood_gender_fig = px.bar(blood_gender_distribution, x='Blood Type', y='Count', color='Gender', barmode='group',
                                  title='Patient Distribution by Blood Type and Gender',
                                  labels={'Count': 'Number of Patients', 'Blood Type': 'Blood Type', 'Gender': 'Gender'},
                                  color_discrete_sequence=new_palette)
        blood_gender_fig.update_layout(title_font_size=24)
        blood_gender_fig.show()

    elif visualization_option == 6:
        # Patient Distribution by Admission Type and Gender
        admission_gender_distribution = healthcare_data.groupby(['Admission Type', 'Gender']).size().reset_index(name='Count')
        admission_gender_fig = px.bar(admission_gender_distribution, x='Admission Type', y='Count', color='Gender', barmode='group',
                                      title='Patient Distribution by Admission Type and Gender',
                                      labels={'Count': 'Number of Patients', 'Admission Type': 'Admission Type', 'Gender': 'Gender'},
                                      color_discrete_sequence=new_palette)
        admission_gender_fig.update_layout(title_font_size=24)
        admission_gender_fig.show()

    elif visualization_option == 7:
        # Patient Distribution by Admission Type and Medical Condition
        admission_condition_distribution = healthcare_data.groupby(['Admission Type', 'Medical Condition']).size().reset_index(name='Count')
        admission_condition_fig = px.bar(admission_condition_distribution, x='Admission Type', y='Count', color='Medical Condition', barmode='group',
                                         title='Patient Distribution by Admission Type and Medical Condition',
                                         labels={'Count': 'Number of Patients', 'Admission Type': 'Admission Type', 'Medical Condition': 'Condition'},
                                         color_discrete_sequence=new_palette)
        admission_condition_fig.update_layout(title_font_size=24)
        admission_condition_fig.show()

    elif visualization_option == 8:
        # Distribution of Test Results by Admission Type
        test_admission_distribution = healthcare_data.groupby(['Test Results', 'Admission Type']).size().reset_index(name='Count')
        test_admission_fig = px.bar(test_admission_distribution, x='Test Results', y='Count', color='Admission Type', barmode='group',
                                    title='Distribution of Test Results by Admission Type',
                                    labels={'Count': 'Number of Tests', 'Test Results': 'Test Results', 'Admission Type': 'Admission Type'},
                                    color_discrete_sequence=new_palette)
        test_admission_fig.update_layout(title_font_size=24)
        test_admission_fig.show()

    elif visualization_option == 9:
        # Medication Distribution by Gender
        medication_gender_distribution = healthcare_data.groupby(['Medication', 'Gender']).size().reset_index(name='Count')
        med_gender_fig = px.bar(medication_gender_distribution, x='Medication', y='Count', color='Gender', barmode='group',
                                title='Medication Distribution by Gender',
                                labels={'Count': 'Number of Prescriptions', 'Medication': 'Medication', 'Gender': 'Gender'},
                                color_discrete_sequence=new_palette)
        med_gender_fig.update_layout(title_font_size=24)
        med_gender_fig.show()
# Function to perform data preprocessing
def preprocess_data():
    global healthcare_data
    if healthcare_data is not None:
        features = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Insurance Provider', 'Medication', 'Test Results']
        target = 'Medical Condition'

        label_encoders = {}
        for col in features:
            if healthcare_data[col].dtype == 'object':
                le = LabelEncoder()
                healthcare_data[col] = le.fit_transform(healthcare_data[col])
                label_encoders[col] = le

        # Define features (X) and target (y)
        global X, y, X_train, X_test, y_train, y_test
        X = healthcare_data[features]
        y = healthcare_data[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Standardize the feature set
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        messagebox.showinfo("Success", "Data Preprocessing Completed!")

# Function to train and evaluate Logistic Regression model
def train_logistic_regression():
    if X_train is not None:
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train, y_train)

        y_pred_log_reg = log_reg.predict(X_test)
        accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

        print("Logistic Regression Accuracy:", accuracy_log_reg)
        print(f'Classification Report:\n{classification_report(y_test, y_pred_log_reg)}')

        conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix for Logistic Regression', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.show()

# Function to train and evaluate Random Forest model
def train_random_forest():
    if X_train is not None:
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)

        y_pred_rf = rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)

        print(f"Random Forest Accuracy: {accuracy_rf}")
        print(f'Classification Report:\n{classification_report(y_test, y_pred_rf)}')

        conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='inferno', cbar=False)
        plt.title('Confusion Matrix for Random Forest', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.show()

# Function to train and evaluate SVM model
def train_svm():
    if X_train is not None:
        svm = SVC(random_state=42)
        svm.fit(X_train, y_train)

        y_pred_svm = svm.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)

        print("Accuracy of SVM model:", accuracy_svm)
        print(f'Classification Report:\n{classification_report(y_test, y_pred_svm)}')

        conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Reds', cbar=False)
        plt.title('Confusion Matrix for Support Vector Machine (SVM)', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.show()

# Function to plot accuracy comparison
def plot_accuracy_comparison():
    global accuracy_log_reg, accuracy_rf, accuracy_svm
    model_accuracies = {
        'Logistic Regression': accuracy_log_reg,
        'Random Forest': accuracy_rf,
        'Support Vector Machine (SVM)': accuracy_svm
    }

    accuracy_df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=accuracy_df, palette='viridis')
    plt.title('Comparison of Model Accuracies', fontsize=18)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0, 1)
    plt.show()

# Buttons for different functionalities
Button(root, text="Load Data", command=load_data).pack(pady=10)
Button(root, text="Show Data Head", command=show_head).pack(pady=10)
Button(root, text="Show Data Shape", command=show_shape).pack(pady=10)
Button(root, text="Check Missing Values", command=check_missing_values).pack(pady=10)
Button(root, text="Standardize Name Column", command=standardize_name).pack(pady=10)
Button(root, text="Convert Dates to Datetime", command=convert_dates).pack(pady=10)
Button(root, text="Perform EDA", command=perform_eda).pack(pady=10)

# Slider to choose visualization
vis_slider = Scale(root, from_=1, to=9, orient=tk.HORIZONTAL, label="Choose Visualization")
vis_slider.pack(pady=10)

# Button to display the selected visualization
Button(root, text="Display Visualization", command=plot_visualization).pack(pady=10)
Button(root, text="Preprocess Data", command=preprocess_data).pack(pady=10)
Button(root, text="Train Logistic Regression", command=train_logistic_regression).pack(pady=10)
Button(root, text="Train Random Forest", command=train_random_forest).pack(pady=10)
Button(root, text="Train SVM", command=train_svm).pack(pady=10)
Button(root, text="Compare Model Accuracies", command=plot_accuracy_comparison).pack(pady=10)

# Start the main loop
root.mainloop()

