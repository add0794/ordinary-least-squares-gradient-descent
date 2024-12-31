# import matplotlib.pyplot as plt
# import numpy as np 
# import pandas as pd 
# from scipy.linalg import inv
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import statsmodels.api as sm
# import sys
# import time

# class LinearRegression:
#     def __init__(self, features_yes, features_no, label, cols_yes, cols_no):
#         self.features_yes = features_yes 
#         self.features_no = features_no
#         self.label = label
#         self.cols_yes = cols_yes
#         self.cols_no = cols_no
        
#         # Initialize coefficient comparison DataFrames
#         self.comparison_yes = pd.DataFrame()
#         self.comparison_yes.attrs['title'] = 'Comparison of Extracurricular - Yes Activities'
        
#         self.comparison_no = pd.DataFrame()
#         self.comparison_no.attrs['title'] = 'Comparison of Extracurricular - No Activities'
        
#         self.performance_metrics_yes = pd.DataFrame()
#         self.performance_metrics_yes.attrs['title'] = 'Performance Metrics - Yes Activities'
#         self.performance_metrics_yes.index = ['elapsed time', 'memory usage']

#         self.performance_metrics_no = pd.DataFrame()
#         self.performance_metrics_no.attrs['title'] = 'Performance Metrics - Yes Activities'
#         self.performance_metrics_no.index = ['elapsed time', 'memory usage']

#         # Add debug flag
#         self.debug = True
    
#     def _compute_parameters(self, features, key, cols, method='numpy'):
#         start_time = time.time()
        
#         # Choose computation method
#         if method == 'numpy':
#             beta_encoding = np.linalg.inv(features.T @ features) @ features.T @ self.label
#         elif method == 'scipy':
#             beta_encoding = inv(features.T @ features) @ features.T @ self.label
            
#         beta_series = pd.Series(data=beta_encoding, index=cols)
#         elapsed_time = time.time() - start_time
        
#         # Memory measurements
#         beta_memory = sys.getsizeof(beta_encoding)
#         series_memory = sys.getsizeof(beta_series)
        
#         # Display results
#         print(f"{method.capitalize()} Method:")
#         print(f"Computed beta coefficients in {elapsed_time:.6f} seconds")
#         print(f"Memory usage for parameters of '{key}' variable: {beta_memory} bytes")
#         print(f"Memory usage for '{key}' variable Series: {series_memory} bytes")
#         print(f"Beta coefficients {key}:")
#         print(beta_series)
#         print("\n")
        
#         return beta_series
    
#     def gradient_descent(self, features, key, cols, learning_rate=0.000001, epochs=100, precision=0.00001):
#         """
#         Perform gradient descent optimization with dimension checking
#         """
#         try:
#             # Ensure features and cols match in dimension
#             if self.debug:
#                 print(f"Features shape: {features.shape}")
#                 print(f"Number of columns: {len(cols)}")
#                 print(f"Columns: {cols}")
            
#             # Split the data without the intercept column
#             X = features[:, 1:] if features.shape[1] == len(cols) else features
#             X_train, X_test, y_train, y_test = train_test_split(X, self.label, test_size=0.2, random_state=10)
            
#             scaling = input('Do you want to scale (normalization, standardization, or no)?')
#             start_time = time.time()
            
#             # Handle scaling
#             if scaling.lower() == 'normalization':
#                 scaler = MinMaxScaler()
#             elif scaling.lower() == 'standardization':
#                 scaler = StandardScaler()
#             elif scaling.lower() == 'no':
#                 scaler = None
#             else:
#                 print('Sorry, that is not a valid response.')
#                 return None, None, None
            
#             # Scale features if requested
#             if scaler:
#                 X_train_scaled = scaler.fit_transform(X_train)
#                 X_test_scaled = scaler.transform(X_test)
#                 X_train_augmented = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
#                 X_test_augmented = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))
#             else:
#                 X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
#                 X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
            
#             # Verify dimensions
#             if self.debug:
#                 print(f"X_train_augmented shape: {X_train_augmented.shape}")
#                 print(f"Number of coefficients to estimate: {len(cols)}")
            
#             # Initialize parameters
#             beta = np.zeros(len(cols))
#             guesses = []
#             losses = []
            
#             # Gradient descent iterations
#             for epoch in range(epochs):
#                 predictions = X_train_augmented @ beta
#                 residuals = predictions - y_train
#                 gradient = (2 / len(y_train)) * X_train_augmented.T @ residuals
                
#                 beta = beta - learning_rate * gradient
                
#                 guesses.append(beta.copy())
#                 loss = np.mean(residuals ** 2)
#                 losses.append(loss)
                
#                 if np.max(np.abs(gradient)) < precision:
#                     break
            
#             elapsed_time = time.time() - start_time
#             beta_series = pd.Series(data=beta, index=cols)
            
#             print(f"Gradient Descent ({key}):")
#             print(f"Computed beta coefficients in {elapsed_time:.6f} seconds")
#             print(f'Memory usage for Pandas Series: {sys.getsizeof(beta_series)} bytes')
#             print(beta_series)
#             print("\n")
            
#             return beta_series, guesses, losses
            
#         except Exception as e:
#             print(f"Error in gradient_descent: {str(e)}")
#             print(f"Traceback: {traceback.format_exc()}")
#             raise
    
#     def train_sklearn_models(self, features, key, cols, ridge_alpha=1.0, lasso_alpha=1.0):
#         """Train multiple sklearn models with error handling"""
#         try:
#             results = {}
            
#             # Remove intercept column for sklearn
#             X = features[:, 1:] if features.shape[1] == len(cols) else features
            
#             models = {
#                 'Scikit-Learn OLS': linear_model.LinearRegression(fit_intercept=True),
#                 'Ridge': linear_model.Ridge(alpha=ridge_alpha, fit_intercept=True),
#                 'Lasso': linear_model.Lasso(alpha=lasso_alpha, fit_intercept=True)
#             }
            
#             for name, model in models.items():
#                 start_time = time.time()
#                 model.fit(X, self.label)
                
#                 # Get coefficients including intercept
#                 coefficients = np.insert(model.coef_, 0, model.intercept_)
#                 beta_series = pd.Series(data=coefficients, index=cols)
                
#                 # Calculate metrics
#                 y_pred = model.predict(X)
#                 r2 = model.score(X, self.label)
#                 mse = np.mean((self.label - y_pred) ** 2)
                
#                 elapsed_time = time.time() - start_time
                
#                 results[name] = {
#                     'Coefficients': beta_series,
#                     'R-squared': r2,
#                     'MSE': mse,
#                     'Elapsed_time': elapsed_time
#                 }
                
#                 print(f"\n{name} ({key}):")
#                 print(f"R-squared: {r2:.4f}")
#                 print(f"MSE: {mse:.4f}")
#                 print(f"Elapsed time: {elapsed_time:.6f} seconds")
#                 print("Coefficients:")
#                 print(beta_series)
            
#             return results
            
#         except Exception as e:
#             print(f"Error in train_sklearn_models: {str(e)}")
#             print(f"Traceback: {traceback.format_exc()}")
#             raise
    
#     def fit_numpy(self):
#         """Compute parameters using NumPy method"""
#         self.numpy_yes = self._compute_parameters(self.features_yes, 'yes', self.cols_yes, 'numpy')
#         self.numpy_no = self._compute_parameters(self.features_no, 'no', self.cols_no, 'numpy')
#         return self.numpy_yes, self.numpy_no
    
#     def fit_scipy(self):
#         """Compute parameters using SciPy method"""
#         self.scipy_yes = self._compute_parameters(self.features_yes, 'yes', self.cols_yes, 'scipy')
#         self.scipy_no = self._compute_parameters(self.features_no, 'no', self.cols_no, 'scipy')
#         return self.scipy_yes, self.scipy_no
    
#     def fit_statsmodels(self, features, key):
#         """Compute parameters using statsmodels"""
#         start_time = time.time()
        
#         X_constant = sm.add_constant(features)
#         model = sm.OLS(self.label, X_constant).fit()
#         summary = model.summary()
        
#         elapsed_time = time.time() - start_time
        
#         print(f"Statsmodels Method ({key}):")
#         print(f"Computed beta coefficients in {elapsed_time:.6f} seconds")
#         print(f"Memory usage for: {sys.getsizeof(model)} bytes")
#         print("\nModel Summary:")
#         print(summary)
#         print("\n")
        
#         return model, summary
    
#     def fit_all_statsmodels(self):
#         """Compute statsmodels for both yes and no cases"""
#         self.statsmodels_yes_model, self.statsmodels_yes_summary = self.fit_statsmodels(self.features_yes, 'yes')
#         self.statsmodels_no_model, self.statsmodels_no_summary = self.fit_statsmodels(self.features_no, 'no')
#         return (self.statsmodels_yes_model, self.statsmodels_yes_summary), (self.statsmodels_no_model, self.statsmodels_no_summary)
    
#     def fit_all_methods(self):
#         """Fit all available methods and store results"""
        
#         # 1. Fit traditional methods
#         print("\nFitting NumPy method...")
#         self.fit_numpy()
        
#         print("\nFitting SciPy method...")
#         self.fit_scipy()
        
#         print("\nFitting statsmodels...")
#         self.fit_all_statsmodels()
        
#         # 2. Fit gradient descent
#         print("\nFitting gradient descent for 'yes' case...")
#         self.gradient_yes, self.guesses_yes, self.losses_yes = self.gradient_descent(
#             self.features_yes, 'yes', self.cols_yes)
        
#         print("\nFitting gradient descent for 'no' case...")
#         self.gradient_no, self.guesses_no, self.losses_no = self.gradient_descent(
#             self.features_no, 'no', self.cols_no)
        
#         # 3. Fit sklearn models
#         print("\nFitting sklearn models for 'yes' case...")
#         self.sklearn_results_yes = self.train_sklearn_models(
#             self.features_yes, 'yes', self.cols_yes)
        
#         print("\nFitting sklearn models for 'no' case...")
#         self.sklearn_results_no = self.train_sklearn_models(
#             self.features_no, 'no', self.cols_no)
        
#         # 4. Create final comparison
#         print("\nCreating comparison dataframes...")
#         self._create_comparison_dataframes()
        
#         print("\nAll methods fitted successfully!")

#     def _create_comparison_dataframes(self):
#         """Internal method to create comparison DataFrames for yes and no cases"""
#         # Update yes comparison
#         if hasattr(self, 'numpy_yes'):
#             self.comparison_yes['NumPy'] = self.numpy_yes
#         if hasattr(self, 'scipy_yes'):
#             self.comparison_yes['SciPy'] = self.scipy_yes
#         if hasattr(self, 'statsmodels_yes_model'):
#             self.comparison_yes['StatsModels'] = self.statsmodels_yes_model.params.values
#         if hasattr(self, 'gradient_yes'):
#             self.comparison_yes['Gradient Descent'] = self.gradient_yes
#         if hasattr(self, 'sklearn_results_yes'):
#             for name, result in self.sklearn_results_yes.items():
#                 self.comparison_yes[name] = result['Coefficients']
                
#         # Update no comparison
#         if hasattr(self, 'numpy_no'):
#             self.comparison_no['NumPy'] = self.numpy_no
#         if hasattr(self, 'scipy_no'):
#             self.comparison_no['SciPy'] = self.scipy_no
#         if hasattr(self, 'statsmodels_no_model'):
#             self.comparison_no['StatsModels'] = self.statsmodels_no_model.params.values
#         if hasattr(self, 'gradient_no'):
#             self.comparison_no['Gradient Descent'] = self.gradient_no
#         if hasattr(self, 'sklearn_results_no'):
#             for name, result in self.sklearn_results_no.items():
#                 self.comparison_no[name] = result['Coefficients']
        
#         return self.comparison_yes, self.comparison_no

#     def create_comparison_dataframes(self):
#         """Public method to create comparison DataFrames, ensuring all methods are fitted"""
#         if not all(hasattr(self, attr) for attr in ['numpy_yes', 'scipy_yes', 'statsmodels_yes_model', 
#                                                 'gradient_yes', 'sklearn_results_yes']):
#             print("Some methods haven't been fitted yet. Running all methods...")
#             self.fit_all_methods()
#         else:
#             self._create_comparison_dataframes()
        
#         return self.comparison_yes, self.comparison_no


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from scipy.linalg import inv
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
import sys
import time

class LinearRegression:
    def __init__(self, features_yes, features_no, label, cols_yes, cols_no):
        self.features_yes = features_yes 
        self.features_no = features_no
        self.label = label
        self.cols_yes = cols_yes
        self.cols_no = cols_no

        # Initialize coefficient comparison DataFrames
        self.comparison_yes = pd.DataFrame()
        self.comparison_yes.attrs['title'] = 'Comparison of Extracurricular - Yes Activities'
        
        self.comparison_no = pd.DataFrame()
        self.comparison_no.attrs['title'] = 'Comparison of Extracurricular - No Activities'
        
        self.performance_metrics_yes = {}
        self.performance_metrics_no = {}

        # Add debug flag
        self.debug = True
    
    def _get_case_data(self, case):
        """
        Retrieve features, label, and columns based on the specified case.
        """
        if case == 'yes':
            return self.features_yes, self.label, self.cols_yes
        elif case == 'no':
            return self.features_no, self.label, self.cols_no
        else:
            raise ValueError("Case must be 'yes' or 'no'")

    def _fit(self, case, method):
        """
        Generalized fit method for computing parameters using specified method.
        """
        features, label, cols = self._get_case_data(case)
        
        # print(f"This is a {method} summary for {case.capitalize()} case!")

        # Start time measurement
        start_time = time.time()

        # Compute beta coefficients based on the chosen method
        if method == 'numpy':
            beta_encoding = np.linalg.inv(features.T @ features) @ features.T @ label
        elif method == 'scipy':
            beta_encoding = inv(features.T @ features) @ features.T @ label
        elif method == 'statsmodels':
            X_constant = sm.add_constant(features)
            self.model = sm.OLS(label, X_constant).fit()
            beta_encoding = self.model.params.values
            # beta_series = pd.Series(data=beta_encoding, index=self.model.params.index)
        else:
            raise ValueError("Method must be {method}.")
        
        beta_series = pd.Series(data=beta_encoding, index=cols)

        # Measure elapsed time
        elapsed_time = time.time() - start_time

        # Measure memory usage
        beta_memory = sys.getsizeof(beta_encoding)
        series_memory = sys.getsizeof(beta_series)
        total_memory = beta_memory + series_memory

        # Display results
        # print(f"Elapsed Time: {elapsed_time:.6f} seconds")
        # print(f"Memory Usage: {total_memory} bytes (Beta: {beta_memory} bytes, Series: {series_memory} bytes)")
        # print(f"Beta Coefficients:\n{beta_series}\n")
        
        return elapsed_time, total_memory, beta_series
        
    # Wrapper for NumPy
    def fit_numpy(self, case):
        return self._fit(case=case, method='numpy')

    # Wrapper for SciPy
    def fit_scipy(self, case):
        return self._fit(case=case, method='scipy')

    # Wrapper for Statsmodels
    def fit_statsmodels(self, case):
        return self._fit(case=case, method='statsmodels')

    def gradient_descent(self, case, learning_rate, epochs, precision):
        """
        Perform gradient descent optimization with dimension checking
        """       
        if case == 'yes':
            features = self.features_yes
            label = self.label
            cols = self.cols_yes
        elif case == 'no':
            features = self.features_no
            label = self.label
            cols = self.cols_no

        # Split the data without the intercept column
        X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=10)
        
        scaling = input('Do you want to scale (normalization, standardization, or no)?')
        start_time = time.time()
        
        # Handle scaling
        if scaling.lower() == 'normalization':
            scaler = MinMaxScaler()
        elif scaling.lower() == 'standardization':
            scaler = StandardScaler()
        elif scaling.lower() == 'no':
            scaler = None
        else:
            print('Sorry, that is not a valid response.')
            return None, None, None
        
        # Scale features if requested
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
                
        # Initialize parameters
        beta = np.zeros(len(cols))
        guesses = []
        losses = []
        
        # Gradient descent iterations
        for epoch in range(epochs):
            predictions = X_train @ beta
            residuals = predictions - y_train
            gradient = (2 / len(y_train)) * X_train.T @ residuals
            
            beta = beta - learning_rate * gradient
            
            guesses.append(beta.copy())
            loss = np.mean(residuals ** 2)
            losses.append(loss)
            
            if np.max(np.abs(gradient)) < precision:
                break
        
        elapsed_time = time.time() - start_time
        beta_series = pd.Series(data=beta, index=cols)
        
        # print(f"Gradient Descent ({key}):")
        print(f"Computed beta coefficients in {elapsed_time:.6f} seconds")
        print(f'Memory usage for Pandas Series: {sys.getsizeof(beta_series)} bytes')
        print(beta_series)
        print("\n")
        
        return beta_series, guesses, losses

    def train_sklearn_models(self, case, ridge_alpha, lasso_alpha):
        """Train multiple sklearn models with error handling"""
        try:
            results = {}
            
            # Remove intercept column for sklearn
            X = features[:, 1:] if features.shape[1] == len(cols) else features
            
            models = {
                'Scikit-Learn OLS': linear_model.LinearRegression(fit_intercept=True),
                'Ridge': linear_model.Ridge(alpha=ridge_alpha, fit_intercept=True),
                'Lasso': linear_model.Lasso(alpha=lasso_alpha, fit_intercept=True)
            }
            
            for name, model in models.items():
                start_time = time.time()
                model.fit(X, self.label)
                
                # Get coefficients including intercept
                coefficients = np.insert(model.coef_, 0, model.intercept_)
                beta_series = pd.Series(data=coefficients, index=cols)
                
                # Calculate metrics
                y_pred = model.predict(X)
                r2 = model.score(X, self.label)
                mse = np.mean((self.label - y_pred) ** 2)
                
                elapsed_time = time.time() - start_time
                
                results[name] = {
                    'Coefficients': beta_series,
                    'R-squared': r2,
                    'MSE': mse,
                    'Elapsed_time': elapsed_time
                }
                
                print(f"\n{name} ({key}):")
                print(f"R-squared: {r2:.4f}")
                print(f"MSE: {mse:.4f}")
                print(f"Elapsed time: {elapsed_time:.6f} seconds")
                print("Coefficients:")
                print(beta_series)
            
            return results
            
        except Exception as e:
            print(f"Error in train_sklearn_models: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")


















    def _create_comparison_dataframes(self):
        """Internal method to create comparison DataFrames for yes and no cases"""
        # Update yes comparison
        if hasattr(self, 'numpy_yes'):
            self.comparison_yes['NumPy'] = self.numpy_yes
        if hasattr(self, 'scipy_yes'):
            self.comparison_yes['SciPy'] = self.scipy_yes
        if hasattr(self, 'statsmodels_yes_model'):
            self.comparison_yes['StatsModels'] = self.statsmodels_yes_model.params.values
        if hasattr(self, 'gradient_yes'):
            self.comparison_yes['Gradient Descent'] = self.gradient_yes
        if hasattr(self, 'sklearn_results_yes'):
            for name, result in self.sklearn_results_yes.items():
                self.comparison_yes[name] = result['Coefficients']
                
        # Update no comparison
        if hasattr(self, 'numpy_no'):
            self.comparison_no['NumPy'] = self.numpy_no
        if hasattr(self, 'scipy_no'):
            self.comparison_no['SciPy'] = self.scipy_no
        if hasattr(self, 'statsmodels_no_model'):
            self.comparison_no['StatsModels'] = self.statsmodels_no_model.params.values
        if hasattr(self, 'gradient_no'):
            self.comparison_no['Gradient Descent'] = self.gradient_no
        if hasattr(self, 'sklearn_results_no'):
            for name, result in self.sklearn_results_no.items():
                self.comparison_no[name] = result['Coefficients']
        
        return self.comparison_yes, self.comparison_no

    def create_comparison_dataframes(self):
        """Public method to create comparison DataFrames, ensuring all methods are fitted"""
        if not all(hasattr(self, attr) for attr in ['numpy_yes', 'scipy_yes', 'statsmodels_yes_model', 
                'gradient_yes', 'sklearn_results_yes']):
            print("Some methods haven't been fitted yet. Running all methods...")
            self.fit_all_methods()
        else:
            self._create_comparison_dataframes()
        
        return self.comparison_yes, self.comparison_no

    def fit_all_methods(self):
        """Fit all available methods and store results"""
        
        # 1. Fit traditional methods
        print("\nFitting NumPy method...")
        self.fit_numpy()
        
        print("\nFitting SciPy method...")
        self.fit_scipy()
        
        print("\nFitting statsmodels...")
        self.fit_all_statsmodels()
        
        # 2. Fit gradient descent
        print("\nFitting gradient descent for 'yes' case...")
        self.gradient_yes, self.guesses_yes, self.losses_yes = self.gradient_descent(
            self.features_yes, 'yes', self.cols_yes)
        
        print("\nFitting gradient descent for 'no' case...")
        self.gradient_no, self.guesses_no, self.losses_no = self.gradient_descent(
            self.features_no, 'no', self.cols_no)
        
        # 3. Fit sklearn models
        print("\nFitting sklearn models for 'yes' case...")
        self.sklearn_results_yes = self.train_sklearn_models(
            self.features_yes, 'yes', self.cols_yes)
        
        print("\nFitting sklearn models for 'no' case...")
        self.sklearn_results_no = self.train_sklearn_models(
            self.features_no, 'no', self.cols_no)
        
        # 4. Create final comparison
        print("\nCreating comparison dataframes...")
        self._create_comparison_dataframes()
        
        print("\nAll methods fitted successfully!")

    # def _create_comparison_dataframes(self):
    #     """Internal method to create comparison DataFrames for yes and no cases"""
    #     # Update yes comparison
    #     if hasattr(self, 'numpy_yes'):
    #         self.comparison_yes['NumPy'] = self.numpy_yes
    #     if hasattr(self, 'scipy_yes'):
    #         self.comparison_yes['SciPy'] = self.scipy_yes
    #     if hasattr(self, 'statsmodels_yes'):
    #         self.comparison_yes['Statsmodels'] = self.statsmodels_yes
        
    #     # Update no comparison
    #     if hasattr(self, 'numpy_no'):
    #         self.comparison_no['NumPy'] = self.numpy_no
    #     if hasattr(self, 'scipy_no'):
    #         self.comparison_no['SciPy'] = self.scipy_no
    #     if hasattr(self, 'statsmodels_no'):
    #         self.comparison_no['Statsmodels'] = self.statsmodels_no
        
    #     return self.comparison_yes, self.comparison_no

    # def fit_all_methods(self):
    #     """Fit all available methods and store results"""
    #     print("\nCreating comparison dataframes...")
    #     self._create_comparison_dataframes()
        
    #     print("\nAll methods fitted successfully!")
