import pandas as pd
from oop_methods import DataLoader, LinearRegression, Visualizer

# Initialize loader object, with the target (label) column and categorical variable(s) to be dummy coded
loader = DataLoader(target_column="Performance Index", categorical_columns=["Extracurricular Activities"])

# Specify where data should be downloaded from
loader = loader.load_data("nikhil7280/student-performance-multiple-linear-regression", "Student_Performance.csv")

# Initialize the Visualizer class and plot different visualizations
visualizer = Visualizer()

processed_data = loader.processed_data
label = loader.target_data

# Correlation matrix
visualizer.plot_correlation_matrix(processed_data)

# Histograms for all numeric columns
# visualizer.plot_histograms(processed_data)

# Distribution of the target variable
data_with_label = pd.concat([processed_data, label], axis=1)
visualizer.plot_target_distribution(data=data_with_label, target_column=loader.target_column)

split_data = loader.create_yes_no_datasets_with_intercept("Extracurricular Activities")

# Split data for categorical variable: "Yes" and "No" for extracurriculars
yes_X = split_data['Yes']['X']
yes_cols = split_data['Yes']['columns']
no_X = split_data['No']['X']
no_cols = split_data['No']['columns']

lr_model = LinearRegression(features_yes=yes_X, features_no=no_X, label=label, cols_yes=yes_cols, cols_no=no_cols)

# cases = ['no', 'yes']
# results = {}  # Store results for both cases

# for case in cases:
#     results_df = pd.DataFrame()
#     coefficients_df = pd.DataFrame()
#     for method in ['numpy', 'scipy', 'statsmodels', 'scikit-learn', 'Ridge', 'Lasso']:
#         if method == 'numpy':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_numpy(case)
#         elif method == 'scipy':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_scipy(case)
#         elif method == 'statsmodels':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case)
#         elif method == 'scikit-learn':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case)
#         elif method == 'Ridge':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case)
#         elif method == 'Lasso', ridge_alpha=1.0, lasso_alpha=0.1:
#             elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case)
#         results_df = pd.concat([results_df,
#                                 pd.DataFrame({'Method': [method],
#                                             'Elapsed Time': [elapsed_time],
#                                             'Memory Usage': [memory_usage]})],
#                             axis=0,
#                             ignore_index=True)
#         coefficients_df = pd.concat([coefficients_df,
#                                     pd.DataFrame({f'{method}_{case}': coefficients}, index= (yes_cols if case == 'yes' else no_cols))],
#                                     axis=1)

#     results[case] = {'results_df': results_df, 'coefficients_df': coefficients_df}

# # Print results neatly after the loop
# for case in cases:
#     print(f"\nResults for Extracurricular Activities: {case}")
#     print(results[case]['results_df'])
#     print(results[case]['coefficients_df'])


cases = ['no', 'yes']
results = {}  # Store results for both cases

# Define Ridge and Lasso alpha values
ridge_alpha = 100.0
lasso_alpha = 100.0

# for case in cases:
#     results_df = pd.DataFrame()
#     coefficients_df = pd.DataFrame()
    
#     for method in ['numpy', 'scipy', 'statsmodels', 'scikit-learn', 'Ridge', 'Lasso']:
#         if method == 'numpy':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_numpy(case)
#         elif method == 'scipy':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_scipy(case)
#         elif method == 'statsmodels':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case)
#         elif method == 'scokit-learn':
#             elapsed_time, memory_usage, coefficients = lr_model.fit_sklearn(case, ridge_alpha=ridge_alpha, lasso_alpha=lasso_alpha)

#         # Record results
#         results_df = pd.concat(
#             [
#                 results_df,
#                 pd.DataFrame(
#                     {'Method': [method], 'Elapsed Time': [elapsed_time], 'Memory Usage': [memory_usage]}
#                 )
#             ],
#             axis=0,
#             ignore_index=True
#         )
#         coefficients_df = pd.concat(
#             [
#                 coefficients_df,
#                 pd.DataFrame(
#                     {f'{method}_{case}': coefficients},
#                     index=(yes_cols if case == 'yes' else no_cols)
#                 )
#             ],
#             axis=1
#         )

#     # Store results for the case
#     results[case] = {'results_df': results_df, 'coefficients_df': coefficients_df}

# # Print results neatly after the loop
# for case in cases:
#     print(f"\nResults for Extracurricular Activities: {case}")
#     print(results[case]['results_df'])
#     print(results[case]['coefficients_df'])

cases = ['no', 'yes']
results = {}  # Store results for both cases

# Define Ridge and Lasso alpha values
ridge_alpha = 100.0
lasso_alpha = 100.0

for case in cases:
    results_df = pd.DataFrame()
    coefficients_df = pd.DataFrame()
    
    for method in ['numpy', 'scipy', 'statsmodels', 'scikit-learn', 'ridge', 'lasso']:
        if method == 'numpy':
            elapsed_time, memory_usage, coefficients = lr_model.fit_numpy(case)
        elif method == 'scipy':
            elapsed_time, memory_usage, coefficients = lr_model.fit_scipy(case)
        elif method == 'statsmodels':
            elapsed_time, memory_usage, coefficients = lr_model.fit_statsmodels(case)
        elif method == 'scikit-learn':
            elapsed_time, memory_usage, coefficients = lr_model.fit_sklearn(case)
        elif method == 'ridge':
            elapsed_time, memory_usage, coefficients = lr_model.fit_ridge(case, ridge_alpha=ridge_alpha)            
        elif method == 'lasso':
            elapsed_time, memory_usage, coefficients = lr_model.fit_lasso(case, lasso_alpha=lasso_alpha)
        else:
            raise Exception("Please enter a correct method.")

        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    {'Method': [method], 'Elapsed Time': [elapsed_time], 'Memory Usage': [memory_usage]}
                )
            ],
            axis=0,
            ignore_index=True
        )
        coefficients_df = pd.concat(
            [
                coefficients_df,
                pd.DataFrame(
                    {f'{method}_{case}': coefficients},
                    index=(yes_cols if case == 'yes' else no_cols)
                )
            ],
            axis=1
        )

    results[case] = {'results_df': results_df, 'coefficients_df': coefficients_df}