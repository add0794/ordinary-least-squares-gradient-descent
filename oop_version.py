import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from ols_gd_OOP import LinearRegression

# Targeting data

def load_and_preprocess_data(path, file):
    """
    Load and preprocess the Student Performance dataset from Kaggle
    """
    # Download dataset
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download('nikhil7280/student-performance-multiple-linear-regression')
    dataset_path = os.path.join(path, file)
    
    # Load raw data
    print("Loading dataset...")
    raw_data = pd.read_csv(dataset_path)
    print("Initial shape:", raw_data.shape)
    
    # Data cleaning
    print("\nCleaning data...")
    # Remove null values
    raw_data.dropna(inplace=True)
    print("Shape after removing null values:", raw_data.shape)
    
    # Remove duplicates
    raw_data.drop_duplicates(inplace=True)
    print("Shape after removing duplicates:", raw_data.shape)
    
    # Create dummy variables
    updated_data = pd.get_dummies(raw_data, columns=['Extracurricular Activities'], dtype=int)
    print("\nFeatures after encoding:", updated_data.columns.tolist())
    
    # Prepare target variable
    y = updated_data['Performance Index']
    
    # Prepare features for 'Yes' case
    X_extracurricular_yes = updated_data.drop(['Performance Index', 'Extracurricular Activities_No'], axis=1)
    X_with_intercept_yes = np.column_stack([np.ones(len(X_extracurricular_yes)), X_extracurricular_yes])
    cols_yes = ['Intercept'] + X_extracurricular_yes.columns.tolist()
    
    # Prepare features for 'No' case
    X_extracurricular_no = updated_data.drop(['Performance Index', 'Extracurricular Activities_Yes'], axis=1)
    X_with_intercept_no = np.column_stack([np.ones(len(X_extracurricular_no)), X_extracurricular_no])
    cols_no = ['Intercept'] + X_extracurricular_no.columns.tolist()
    
    print("\nData preparation complete!")
    
    return {
        'X_with_intercept_yes': X_with_intercept_yes,
        'X_with_intercept_no': X_with_intercept_no,
        'X_extracurricular_yes': X_extracurricular_yes,
        'X_extracurricular_no': X_extracurricular_no,
        'y': y,
        'cols_yes': cols_yes,
        'cols_no': cols_no,
        'raw_data': raw_data,
        'processed_data': updated_data
    }

data = load_and_preprocess_data('nikhil7280/student-performance-multiple-linear-regression', 'Student_Performance.csv')

lr_model = LinearRegression(data['X_with_intercept_yes'], data['X_with_intercept_no'], data['y'], data['cols_yes'], data['cols_no'])

cases = ['no', 'yes']

for case in cases:
    results_df = pd.DataFrame()
    coefficients_df = pd.DataFrame()
    for method in ['numpy', 'scipy', 'statsmodels']: # methods = ['numpy', 'scipy', 'statsmodels', 'gradient descent', 'scikit-learn', 'ridge regression', 'lasso regression']
        if case == 'no':
            elapsed_time, memory_usage, coefficients = lr_model.fit_numpy(case) 
            elapsed_time, memory_usage, coefficients = lr_model.fit_scipy(case) 
            elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case) 
        elif case == 'yes':
            elapsed_time, memory_usage, coefficients = lr_model.fit_numpy(case) 
            elapsed_time, memory_usage, coefficients = lr_model.fit_scipy(case) 
            elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case)         
        results_df = pd.concat([results_df, 
                                pd.DataFrame({'Method': [method], 
                                             'Elapsed Time': [elapsed_time], 
                                             'Memory Usage': [memory_usage]})], 
                               axis=0, 
                               ignore_index=True) 
        coefficients_df = pd.concat([coefficients_df, 
                                    pd.DataFrame({f'{method}_{case}': coefficients})], 
                                    axis=1)
    
    if case == 'no':
        no_extracurriculars_results = results_df
        no_extracurriculars_coefficients = coefficients_df
    elif case == 'yes':
        yes_extracurriculars_results = results_df
        yes_extracurriculars_coefficients = coefficients_df

print(no_extracurriculars_results)
print(no_extracurriculars_coefficients)
print(yes_extracurriculars_results)
print(yes_extracurriculars_coefficients)

# Visualizations

# def plot_data_overview(data_dict):
#     """
#     Create overview plots of the dataset using seaborn's default style
#     """
#     # Set the style using seaborn
#     sns.set_theme()
    
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
#     # Performance Index distribution
#     sns.histplot(data=data_dict['processed_data'], x='Performance Index', ax=axes[0,0])
#     axes[0,0].set_title('Distribution of Performance Index')
    
#     # Performance by Extracurricular Activities
#     sns.boxplot(data=data_dict['raw_data'], x='Extracurricular Activities', y='Performance Index', ax=axes[0,1])
#     axes[0,1].set_title('Performance by Extracurricular Activities')
    
#     # Correlation matrix for numerical features
#     numerical_cols = data_dict['processed_data'].select_dtypes(include=[np.number]).columns
#     correlation_matrix = data_dict['processed_data'][numerical_cols].corr()
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1,0])
#     axes[1,0].set_title('Correlation Matrix')
    
#     # Sample scatter plot
#     sns.scatterplot(data=data_dict['raw_data'], x='Hours Studied', y='Performance Index', ax=axes[1,1])
#     axes[1,1].set_title('Performance vs Hours Studied')
    
#     plt.tight_layout()
#     return fig

# def run_analysis(save_results=True):
#     """
#     Run the complete analysis pipeline
#     """
#     try:
#         # Load and preprocess data
#         print("Starting data preprocessing...")
#         data_dict = load_and_preprocess_data()
        
#         # Create visualization
#         print("\nCreating data visualizations...")
#         fig = plot_data_overview(data_dict)
#         if save_results:
#             os.makedirs('results', exist_ok=True)
#             fig.savefig('results/data_overview.png')
#         plt.close()
        
#         # Initialize the model
#         print("\nInitializing model...")
#         model = LinearRegression(
#             data_dict['X_with_intercept_yes'],
#             data_dict['X_with_intercept_no'],
#             data_dict['y'],
#             data_dict['cols_yes'],
#             data_dict['cols_no']
#         )
#         model.debug = True  # Enable debug output
        
#         # Fit all methods
#         print("\nFitting all regression methods...")
#         model.fit_all_methods()
        
#         # Get comparisons
#         comparison_yes, comparison_no = model.comparison_yes, model.comparison_no
        
#         # Display results
#         print("\nResults for students with extracurricular activities:")
#         print(comparison_yes)
#         print("\nResults for students without extracurricular activities:")
#         print(comparison_no)
        
#         # Save results
#         if save_results:
#             print("\nSaving results...")
#             model.save_results()
            
#         print("\nAnalysis complete!")
#         return model, data_dict
        
#     except Exception as e:
#         print(f"Error during analysis: {str(e)}")
#         raise

# if __name__ == "__main__":
#     try:
#         model, data_dict = run_analysis(save_results=True)
#         print("\nScript executed successfully!")
#     except Exception as e:
#         print(f"Script execution failed: {str(e)}")
#         sys.exit(1)