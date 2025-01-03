import kagglehub
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd 
from scipy.linalg import inv
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
import time

class DataLoader:
    def __init__(self, target_column, categorical_columns=None):
        """
        Initialize the DataLoader with column specifications.
        
        :param target_column: The name of the target variable
        :param categorical_columns: List of categorical column names for one-hot encoding (optional)
        """
        # Column specifications
        self.target_column = target_column
        self.categorical_columns = categorical_columns if categorical_columns else []
        
        # Data containers
        self.raw_data = pd.DataFrame()
        self.processed_data = pd.DataFrame()
        self.target_data = pd.Series(dtype=float)
        self.feature_columns = []

    def load_data(self, kaggle, file):
        """Load and preprocess data with improved error handling."""
        try:
            print(f"Downloading dataset: {kaggle}...")
            path = kagglehub.dataset_download(kaggle)

            if path is None:  # Check if download was successful
                raise ValueError(f"Failed to download dataset from {kaggle}. Check Kaggle credentials and dataset name.")

            dataset_path = os.path.join(path, file)
            print(f"Dataset downloaded to: {dataset_path}")

            if not os.path.exists(dataset_path):  # Check if file exists
                raise FileNotFoundError(f"File '{file}' not found in downloaded dataset at {path}.")


            print("Loading dataset into memory...")
            self.raw_data = pd.read_csv(dataset_path)
            print(f"Initial dataset shape: {self.raw_data.shape}")

            self._preprocess_data()
            return self # Return self to allow method chaining

        except (ValueError, FileNotFoundError, OSError, pd.errors.ParserError, Exception) as e:  # Catch potential errors
            print(f"Error loading data: {e}")
            return None  # Explicitly return None in case of errors

    def _preprocess_data(self):
        """
        Perform preprocessing, such as removing nulls, duplicates, and encoding categorical variables.
        """
        # Drop nulls and duplicates
        self.raw_data.dropna(inplace=True)
        self.raw_data.drop_duplicates(inplace=True)
        
        # Copy raw data to processed data
        self.processed_data = self.raw_data.copy()
        
        # Extract target variable first
        self.target_data = self.processed_data.pop(self.target_column)
        
        # Perform dummy coding if categorical columns exist
        if self.categorical_columns:
            self._dummy_coding()
        
        # Update feature columns
        self.feature_columns = self.processed_data.columns.tolist()

    def _dummy_coding(self):
        """
        Perform one-hot encoding on categorical columns if specified.
        Updates processed_data and feature_columns.
        """
        if not len(self.raw_data):
            return
            
        if self.categorical_columns:
            self.processed_data = pd.get_dummies(
                self.raw_data.drop(columns=[self.target_column]), 
                columns=self.categorical_columns, 
                dtype=int
            )
        else:
            self.processed_data = self.raw_data.drop(columns=[self.target_column]).copy()
            
        self.feature_columns = self.processed_data.columns.tolist()
        self.target_data = self.raw_data[self.target_column]

    def create_yes_no_datasets(self, base_column_name):
        """
        Creates "Yes" and "No" DataFrames, dropping the opposite dummy column.

        Args:
            base_column_name: The base name of the categorical variable.

        Returns:
            A dictionary with 'Yes' and 'No' DataFrames, or None on error.
        """
        try:
            dummy_cols = [col for col in self.processed_data.columns if col.startswith(base_column_name + "_")]
            if not dummy_cols:
                raise KeyError(f"No dummy columns found for base name: {base_column_name}")

            yes_col = [col for col in dummy_cols if col.endswith("_Yes") or col.endswith("_yes")][0]
            no_col = [col for col in dummy_cols if col.endswith("_No") or col.endswith("_no")][0]
            
            # Create DataFrames and drop the opposite column
            yes_df = self.processed_data.drop(columns=no_col).copy()
            no_df = self.processed_data.drop(columns=yes_col).copy()

            return {'Yes': yes_df, 'No': no_df}

        except KeyError as e:
            print(f"KeyError: {e}")
            return None
        except IndexError as e:
            print(f"IndexError: {e}. Ensure that the categories Yes and No exist")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def create_yes_no_datasets_with_intercept(self, base_column_name):
        """
        Creates "Yes" and "No" DataFrames with intercept and returns them as numpy arrays.

        Args:
            base_column_name: The base name of the categorical variable.

        Returns:
            A dictionary with keys 'Yes' and 'No', each containing a dictionary with 'X' (NumPy array with intercept) and 'columns' (list of column names).
            Returns None if there is an error
        """
        try:
            dummy_cols = [col for col in self.processed_data.columns if col.startswith(base_column_name + "_")]
            if not dummy_cols:
                raise KeyError(f"No dummy columns found for base name: {base_column_name}")

            yes_col = [col for col in dummy_cols if col.endswith("_Yes") or col.endswith("_yes")][0]
            no_col = [col for col in dummy_cols if col.endswith("_No") or col.endswith("_no")][0]
            
            # Create DataFrames and drop the opposite column
            yes_df = self.processed_data.drop(columns=no_col).copy()
            no_df = self.processed_data.drop(columns=yes_col).copy()

            X_with_intercept_yes = np.column_stack([np.ones(len(yes_df)), yes_df])
            cols_yes = ['Intercept'] + yes_df.columns.tolist()

            X_with_intercept_no = np.column_stack([np.ones(len(no_df)), no_df])
            cols_no = ['Intercept'] + no_df.columns.tolist()

            return {'Yes': {'X': X_with_intercept_yes, 'columns': cols_yes},
                    'No': {'X': X_with_intercept_no, 'columns': cols_no}}

        except KeyError as e:
            print(f"KeyError: {e}")
            return None
        except IndexError as e:
            print(f"IndexError: {e}. Ensure that the categories Yes and No exist")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
            
class Visualizer:
    @staticmethod
    def plot_correlation_matrix(data, target_column):
        """
        Plot a correlation matrix of the data.
        """
        mask = np.zeros_like(data.corr())
        upper_triangle = np.triu_indices_from(mask)
        mask[upper_triangle] = True
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), mask=mask, annot=True).set(title=f'Correlation Matrix of {target_column}')
        plt.show()

    @staticmethod
    def plot_target_distribution(data, target_column):
        """
        Plot the distribution of the target variable.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(x=f'{target_column}', data=data, kde=True, bins=15, color='green', alpha=0.7)
        plt.title(f"Distribution of Target Variable: {target_column}")
        plt.xlabel(target_column)
        plt.ylabel("Frequency")
        plt.show()

class CustomLinearRegression:
    def __init__(self, features_yes, features_no, label, cols_yes, cols_no):
        self.features_yes = features_yes
        self.features_no = features_no
        self.label = label
        self.cols_yes = cols_yes
        self.cols_no = cols_no
        self.models = {}
        self.coefficients = {}

    def _get_case_data(self, case):
        if case == 'yes':
            return self.features_yes, self.label, self.cols_yes
        elif case == 'no':
            return self.features_no, self.label, self.cols_no
        else:
            raise ValueError("Case must be 'yes' or 'no'")

    def _fit(self, case, method, ridge_alpha=None, lasso_alpha=None, epochs=None, learning_rate=None, precision=None):
        """
        Fits a linear regression model using the specified method.

        Args:
            case: The case to use ('yes' or 'no').
            method: The regression method to use ('numpy', 'scipy', 'statsmodels', 
                    'scikit-learn', 'ridge', or 'lasso').
            ridge_alpha: The alpha parameter for Ridge regression.
            lasso_alpha: The alpha parameter for Lasso regression.

        Returns:
            A tuple containing the elapsed time, memory usage (placeholder), 
            and a pandas Series of the estimated coefficients.
        """
        features, label, cols = self._get_case_data(case)
        start_time = time.time()

        if method == 'numpy':
            beta_encoding = np.linalg.inv(features.T @ features) @ features.T @ label
        elif method == 'scipy':
            beta_encoding = inv(features.T @ features) @ features.T @ label
        elif method == 'statsmodels':
            X_constant = sm.add_constant(features)
            self.model = sm.OLS(label, X_constant).fit()
            beta_encoding = self.model.params.values
        elif method == 'scikit-learn':
            X = features[:, 1:] if features.shape[1] == len(cols) else features
            model = LinearRegression()
            model.fit(X, label)
            beta_encoding = np.insert(model.coef_, 0, model.intercept_)
            self.models["scikit-learn"] = model
            self.coefficients["scikit-learn"] = beta_encoding
        elif method == 'ridge':
            if ridge_alpha is None:
                raise ValueError("ridge_alpha must be provided for Ridge regression")
            X = features[:, 1:] if features.shape[1] == len(cols) else features
            model = Ridge(alpha=ridge_alpha)
            model.fit(X, label)
            beta_encoding = np.insert(model.coef_, 0, model.intercept_)
            self.models["Ridge"] = model
            self.coefficients["Ridge"] = beta_encoding
        elif method == 'lasso':
            if lasso_alpha is None:
                raise ValueError("lasso_alpha must be provided for Lasso regression")
            X = features[:, 1:] if features.shape[1] == len(cols) else features
            model = Lasso(alpha=lasso_alpha)
            model.fit(X, label)
            beta_encoding = np.insert(model.coef_, 0, model.intercept_)
            self.models["Lasso"] = model
            self.coefficients["Lasso"] = beta_encoding
        elif method == 'gradient descent':
            # Split the data without the intercept column
            X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=10)
            
            scaling = input('Do you want to scale (normalization, standardization, or no)? ')
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
            beta_encoding = beta.copy()  # Ensure beta_encoding is always defined
            
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
                    beta_encoding = beta
                    break
            else:
                # Assign the final beta to beta_encoding if the loop completes without breaking
                beta_encoding = beta
        else:
            raise ValueError(f"Invalid method: {method}")

        elapsed_time = time.time() - start_time
        beta_series = pd.Series(data=beta_encoding, index=cols)
        
        # Return with placeholder memory usage that will be replaced by decorator
        return elapsed_time, beta_series

    def fit_numpy(self, case):
        return self._fit(case=case, method='numpy')

    def fit_scipy(self, case):
        return self._fit(case=case, method='scipy')

    def fit_statsmodels(self, case):
        return self._fit(case=case, method='statsmodels')

    def fit_sklearn(self, case, ridge_alpha=None, lasso_alpha=None):
        return self._fit(case=case, method='scikit-learn', ridge_alpha=ridge_alpha, lasso_alpha=lasso_alpha)

    def fit_ridge(self, case, ridge_alpha):
        return self._fit(case=case, method='ridge', ridge_alpha=ridge_alpha)

    def fit_lasso(self, case, lasso_alpha):
        return self._fit(case=case, method='lasso', lasso_alpha=lasso_alpha)   
    
    def fit_gradient_descent(self, case, epochs, learning_rate, precision):  
        return self._fit(case=case, method='gradient descent', epochs=epochs, learning_rate=learning_rate, precision=precision)