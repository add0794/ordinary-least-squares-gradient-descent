import pandas as pd
from oop_methods import DataLoader, LinearRegression, Visualizer

loader = DataLoader(target_column="Performance Index", categorical_columns=["Extracurricular Activities"])
loader = loader.load_data("nikhil7280/student-performance-multiple-linear-regression", "Student_Performance.csv")

label = loader.target_data

# Initialize the Visualizer class and plot different visualizations
visualizer = Visualizer()

processed_data = loader.processed_data

# Correlation matrix
visualizer.plot_correlation_matrix(processed_data)

# Histograms for all numeric columns
visualizer.plot_histograms(processed_data)

# Scatter matrix for selected columns
columns_to_plot = ['Feature1', 'Feature2', 'Feature3']
visualizer.plot_scatter_matrix(processed_data, columns=columns_to_plot)

# Box plots for all numeric columns
visualizer.plot_box_plots(processed_data)

# Distribution of the target variable
visualizer.plot_target_distribution(processed_data, target_column='Performance Index')

split_data = loader.create_yes_no_datasets_with_intercept("Extracurricular Activities")

# if split_data:
yes_X = split_data['Yes']['X']
yes_cols = split_data['Yes']['columns']
no_X = split_data['No']['X']
no_cols = split_data['No']['columns']

lr_model = LinearRegression(features_yes=yes_X, features_no=no_X, label=label, cols_yes=yes_cols, cols_no=no_cols)

cases = ['no', 'yes']
results = {}  # Store results for both cases

for case in cases:
    results_df = pd.DataFrame()
    coefficients_df = pd.DataFrame()
    for method in ['numpy', 'scipy', 'statsmodels']:
        if method == 'numpy':
            elapsed_time, memory_usage, coefficients = lr_model.fit_numpy(case)
        elif method == 'scipy':
            elapsed_time, memory_usage, coefficients = lr_model.fit_scipy(case)
        elif method == 'statsmodels':
            elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case)

        results_df = pd.concat([results_df,
                                pd.DataFrame({'Method': [method],
                                            'Elapsed Time': [elapsed_time],
                                            'Memory Usage': [memory_usage]})],
                            axis=0,
                            ignore_index=True)
        coefficients_df = pd.concat([coefficients_df,
                                    pd.DataFrame({f'{method}_{case}': coefficients}, index= (yes_cols if case == 'yes' else no_cols))],
                                    axis=1)

    results[case] = {'results_df': results_df, 'coefficients_df': coefficients_df}

# Print results neatly after the loop
for case in cases:
    print(f"\nResults for Extracurricular Activities: {case}")
    print(results[case]['results_df'])
    print(results[case]['coefficients_df'])

#     else:
#         print("Error creating yes no dataframes with intercept from kaggle data")
# else:
#     print("Error loading data from kaggle")




# Initialize with target and categorical column definitions
# data_loader = DataLoader(target_column="Performance Index", categorical_columns=["Extracurricular Activities"])

# # Load and preprocess data later
# data_loader.load_data("nikhil7280/student-performance-multiple-linear-regression", "Student_Performance.csv")

# data_loader._preprocess_data()

# # Access processed features and target
# print(data_loader.processed_data)
# print(data_loader.target_data)

# Initialize the DataLoader
# loader = DataLoader(
#     target_column='Performance Index',
#     categorical_columns=['Extracurricular Activities']
# )
# # loader.dummy_coding()

# data = loader.load_data(kaggle='nikhil7280/student-performance-multiple-linear-regression', file='Student_Performance.csv')

# import pandas as pd
# import numpy as np
# import kagglehub
# import os

# # ... (DataLoader class as provided in your last message)

# # Example Usage:
# data = {'Performance Index': [1, 2, 3, 4, 5, 6],
#         'Extracurricular Activities': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
#         'Other_Feature': [4, 5, 6, 7, 8, 9],
#         'Another_Feature': [10, 11, 12, 13, 14, 15]}

# feature_columns = ['Extracurricular Activities', 'Other_Feature', 'Another_Feature']
# df = pd.DataFrame(data)
# processor = DataLoader("Performance Index", ['Extracurricular Activities'])
# processor.raw_data = df
# processor._preprocess_data()

# split_data = processor.create_yes_no_datasets('Extracurricular Activities')

# if split_data:
#     yes_df = split_data['Yes']
#     no_df = split_data['No']

#     print("Yes DataFrame:\n", yes_df)
#     print("No DataFrame:\n", no_df)
# else:
#     print("An error occurred during DataFrame creation.")

# split_data_with_intercept = processor.create_yes_no_datasets_with_intercept('Extracurricular Activities')

# if split_data_with_intercept:
#     yes_X = split_data_with_intercept['Yes']['X']
#     yes_cols = split_data_with_intercept['Yes']['columns']
#     no_X = split_data_with_intercept['No']['X']
#     no_cols = split_data_with_intercept['No']['columns']

#     print("Yes Data with intercept:\n", yes_X)
#     print("Yes Columns:\n", yes_cols)
#     print("No Data with intercept:\n", no_X)
#     print("No Columns:\n", no_cols)
# else:
#     print("An error occurred during DataFrame creation.")

#Example of loading data from Kaggle
# loader = DataLoader(target_column='SalePrice', categorical_columns=['MSZoning', 'Street'])
# loader = loader.load_data('competitions/house-prices-advanced-regression-techniques', 'train.csv')
# loader = DataLoader(target_column="Performance Index", categorical_columns=["Extracurricular Activities"])
# loader = loader.load_data("nikhil7280/student-performance-multiple-linear-regression", "Student_Performance.csv")

# label = loader.target_data

# if loader:
#     split_data_kaggle = loader.create_yes_no_datasets("Extracurricular Activities")
#     if split_data_kaggle:
#         yes_df_kaggle = split_data_kaggle["Yes"]
#         no_df_kaggle = split_data_kaggle["No"]
#         print("Kaggle Yes DataFrame:\n", yes_df_kaggle)
#         print("Kaggle No DataFrame:\n", no_df_kaggle)
#     else:
#         print("Error creating yes no dataframes from kaggle data")

#     split_data_with_intercept_kaggle = loader.create_yes_no_datasets_with_intercept("Extracurricular Activities")
#     if split_data_with_intercept_kaggle:
#         yes_X_kaggle = split_data_with_intercept_kaggle['Yes']['X']
#         yes_cols_kaggle = split_data_with_intercept_kaggle['Yes']['columns']
#         no_X_kaggle = split_data_with_intercept_kaggle['No']['X']
#         no_cols_kaggle = split_data_with_intercept_kaggle['No']['columns']

#         lr_model = LinearRegression(features_yes=yes_X_kaggle, features_no=no_X_kaggle, label=label, cols_yes=yes_cols_kaggle, cols_no=no_cols_kaggle)

#         cases = ['no', 'yes']

#         for case in cases:
#             results_df = pd.DataFrame()
#             coefficients_df = pd.DataFrame()
#             for method in ['numpy', 'scipy', 'statsmodels']: # methods = ['numpy', 'scipy', 'statsmodels', 'gradient descent', 'scikit-learn', 'ridge regression', 'lasso regression']
#                 if case == 'no':
#                     elapsed_time, memory_usage, coefficients = lr_model.fit_numpy(case) 
#                     elapsed_time, memory_usage, coefficients = lr_model.fit_scipy(case) 
#                     elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case) 
#                 elif case == 'yes':
#                     elapsed_time, memory_usage, coefficients = lr_model.fit_numpy(case) 
#                     elapsed_time, memory_usage, coefficients = lr_model.fit_scipy(case) 
#                     elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case)         
#                 results_df = pd.concat([results_df, 
#                                         pd.DataFrame({'Method': [method], 
#                                                     'Elapsed Time': [elapsed_time], 
#                                                     'Memory Usage': [memory_usage]})], 
#                                     axis=0, 
#                                     ignore_index=True) 
#                 coefficients_df = pd.concat([coefficients_df, 
#                                             pd.DataFrame({f'{method}_{case}': coefficients})], 
#                                             axis=1)
            
#             if case == 'no':
#                 no_extracurriculars_results = results_df
#                 no_extracurriculars_coefficients = coefficients_df
#             elif case == 'yes':
#                 yes_extracurriculars_results = results_df
#                 yes_extracurriculars_coefficients = coefficients_df

#             print(no_extracurriculars_results)
#             print(no_extracurriculars_coefficients)
#             print(yes_extracurriculars_results)
#             print(yes_extracurriculars_coefficients)

#     else:
#         print("Error creating yes no dataframes with intercept from kaggle data")
# else:
#     print("Error loading data from kaggle")

# print(data.create_yes_no_datasets('Extracurricular Activities_Yes'))


# import pandas as pd
# import numpy as np

# # ... (DataProcessor class from previous response)

# split_dfs = data.create_split_dataframes(['Extracurricular Activities_Yes', 'Extracurricular Activities_No'])

# if split_dfs:
#     yes_key = 'Extracurricular Activities_Yes'
#     no_key = 'Extracurricular Activities_No'

#     if yes_key in split_dfs and no_key in split_dfs:
#         X_yes_non_zero = np.column_stack([np.ones(len(split_dfs[yes_key]['non_zero'])), split_dfs[yes_key]['non_zero']])
#         X_yes_zero = np.column_stack([np.ones(len(split_dfs[yes_key]['zero'])), split_dfs[yes_key]['zero']])
#         X_no_non_zero = np.column_stack([np.ones(len(split_dfs[no_key]['non_zero'])), split_dfs[no_key]['non_zero']])
#         X_no_zero = np.column_stack([np.ones(len(split_dfs[no_key]['zero'])), split_dfs[no_key]['zero']])

#         print("\nExtracurricular Activities_Yes Non-Zero Array:\n", X_yes_non_zero)
#         print("\nExtracurricular Activities_Yes Zero Array:\n", X_yes_zero)
#         print("\nExtracurricular Activities_No Non-Zero Array:\n", X_no_non_zero)
#         print("\nExtracurricular Activities_No Zero Array:\n", X_no_zero)
#     else:
#         print("One or both keys ('Yes' and 'No') not found in split_dfs.")
# else:
#     print("An error occurred during DataFrame creation.")

# Load and process your data
# data = loader.load_data(kaggle='nikhil7280/student-performance-multiple-linear-regression', file='Student_Performance.csv')

# print(data.target_data)

# # Split the data
# split_data = loader.split_data('Extracurricular Activities')

# print(split_data)

# Access the data for each category
# X_yes = split_data['Yes']['X']
# cols_yes = split_data['Yes']['columns']
# print(X_yes.shape)

# X_no = split_data['No']['X']
# cols_no = split_data['No']['columns']

# label = loader.target_data

# lr_model = LinearRegression(features_yes=X_yes, features_no=X_no, label=label, cols_yes=cols_yes, cols_no=cols_no)

# print(lr_model.X_yes)

# cases = ['no', 'yes']

# for case in cases:
#     results_df = pd.DataFrame()
#     coefficients_df = pd.DataFrame()
#     for method in ['numpy', 'scipy', 'statsmodels']: # methods = ['numpy', 'scipy', 'statsmodels', 'gradient descent', 'scikit-learn', 'ridge regression', 'lasso regression']
#         if case == 'no':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_numpy(case) 
#             elapsed_time, memory_usage, coefficients = lr_model.fit_scipy(case) 
#             elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case) 
#         elif case == 'yes':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_numpy(case) 
#             elapsed_time, memory_usage, coefficients = lr_model.fit_scipy(case) 
#             elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case)         
#         results_df = pd.concat([results_df, 
#                                 pd.DataFrame({'Method': [method], 
#                                              'Elapsed Time': [elapsed_time], 
#                                              'Memory Usage': [memory_usage]})], 
#                                axis=0, 
#                                ignore_index=True) 
#         coefficients_df = pd.concat([coefficients_df, 
#                                     pd.DataFrame({f'{method}_{case}': coefficients})], 
#                                     axis=1)
    
#     if case == 'no':
#         no_extracurriculars_results = results_df
#         no_extracurriculars_coefficients = coefficients_df
#     elif case == 'yes':
#         yes_extracurriculars_results = results_df
#         yes_extracurriculars_coefficients = coefficients_df

# print(no_extracurriculars_results)
# print(no_extracurriculars_coefficients)
# print(yes_extracurriculars_results)
# print(yes_extracurriculars_coefficients)