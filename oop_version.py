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