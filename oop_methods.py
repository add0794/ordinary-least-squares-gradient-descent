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