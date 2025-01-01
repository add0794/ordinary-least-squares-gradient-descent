import kagglehub
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd 
from scipy.linalg import inv
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
import sys
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

    # def create_yes_no_datasets_from_split(self, split_dataframes, base_column_name):
    #     try:
    #         yes_key = [key for key in split_dataframes if key.endswith("_Yes") or key.endswith("_yes")][0]
    #         no_key = [key for key in split_dataframes if key.endswith("_No") or key.endswith("_no")][0]
    #         yes_df = split_dataframes[yes_key]
    #         no_df = split_dataframes[no_key]
    #         return {'Yes': yes_df, 'No': no_df}

    #     except IndexError as e:
    #         print("Could not find the yes or no key")
    #         return None
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         return None

    # def create_yes_no_datasets(self, split_column):
    #     """
    #     Creates "Yes" and "No" DataFrames based on a dummy-coded column,
    #     retaining all other dummy-coded columns in both.

    #     Args:
    #         split_column: The name of the dummy-coded column to split on.

    #     Returns:
    #         A dictionary with keys 'Yes' and 'No', each containing a dictionary with 'X' (NumPy array with intercept) and 'columns' (list of column names).
    #         Returns None if there is an error
    #     """
    #     if split_column not in self.feature_columns:
    #         print(f"Error: Column '{split_column}' not found in feature columns.")
    #         return None

    #     try:
    #         # Create "Yes" DataFrame (where the split column is 1)
    #         yes_df = self.processed_data[self.processed_data[split_column] == 1].copy()
    #         X_with_intercept_yes = np.column_stack([np.ones(len(yes_df)), yes_df])
    #         cols_yes = ['Intercept'] + yes_df.columns.tolist()

    #         # Create "No" DataFrame (where the split column is 0)
    #         no_df = self.processed_data[self.processed_data[split_column] == 0].copy()
    #         X_with_intercept_no = np.column_stack([np.ones(len(no_df)), no_df])
    #         cols_no = ['Intercept'] + no_df.columns.tolist()

    #         return {'Yes': {'X': X_with_intercept_yes, 'columns': cols_yes},
    #                 'No': {'X': X_with_intercept_no, 'columns': cols_no}}

    #     except KeyError as e:
    #         print(f"KeyError: {e}")
    #         return None
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         return None

    # def create_split_dataframes(self, split_column):
    #     """
    #     Creates "Yes" and "No" datasets based on a split column.

    #     Args:
    #         split_column: The name of the categorical column to split on.

    #     Returns:
    #         A dictionary with keys 'Yes' and 'No', each containing a dictionary with 'X' (NumPy array with intercept) and 'columns' (list of column names).
    #         Returns None if there is an error
    #     """
    #     if split_column not in self.feature_columns:
    #         print(f"Error: Column '{split_column}' not found in feature columns.")
    #         return None
        
    #     try:
    #         # Create "Yes" dataset (where the split column is present)
    #         X_yes = self.processed_data.drop(columns=[col for col in self.processed_data.columns if col.startswith(split_column) and col != split_column], errors='ignore') #Drop all other dummy columns
    #         X_with_intercept_yes = np.column_stack([np.ones(len(X_yes)), X_yes])
    #         cols_yes = ['Intercept'] + X_yes.columns.tolist()

    #         # Create "No" dataset (where the split column is absent)
    #         X_no = self.processed_data.drop(columns=[split_column])
    #         X_with_intercept_no = np.column_stack([np.ones(len(X_no)), X_no])
    #         cols_no = ['Intercept'] + X_no.columns.tolist()

    #         return {'Yes': {'X': X_with_intercept_yes, 'columns': cols_yes},
    #                 'No': {'X': X_with_intercept_no, 'columns': cols_no}}
    #     except KeyError as e:
    #         print(f"KeyError: {e}")
    #         return None
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         return None


    # def create_split_dataframes(self, split_columns):
    #     """
    #     Creates new DataFrames based on specified columns.

    #     Args:
    #         split_columns: A list of column names to split on.

    #     Returns:
    #         A dictionary where keys are the column names and values are dictionaries
    #         containing two DataFrames:
    #         - 'non_zero': DataFrame where the column has a non-zero value.
    #         - 'zero': DataFrame where the column has a zero value.
    #         Returns None if there is an error
    #     """

    #     if not isinstance(split_columns, list):
    #         print("Error: split_columns must be a list.")
    #         return None

    #     for split_column in split_columns:
    #         if split_column not in self.feature_columns:
    #             print(f"Error: Column '{split_column}' not found in feature columns.")
    #             return None

    #     try:
    #         split_dataframes = {}
    #         for split_column in split_columns:
    #             non_zero_df = self.processed_data[self.processed_data[split_column] != 0].copy()
    #             zero_df = self.processed_data[self.processed_data[split_column] == 0].copy()

    #             split_dataframes[split_column] = {'non_zero': non_zero_df, 'zero': zero_df}

    #         return split_dataframes

    #     except KeyError:
    #         print(f"KeyError: A column was not found")
    #         return None
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         return None

    # def split_data(self, split_columns):
    #     """
    #     Splits the data based on multiple columns, keeping all rows in each split.

    #     Args:
    #         split_columns: A list of column names to split on.

    #     Returns:
    #         A dictionary where keys are the column names and values are tuples
    #         containing:
    #         - A NumPy array with intercept for rows where that column has a non-zero value.
    #         - A NumPy array with intercept for rows where that column has a zero value.
    #         Returns None if there is an error
    #     """

    #     if not isinstance(split_columns, list):
    #         print("Error: split_columns must be a list.")
    #         return None

    #     for split_column in split_columns:
    #         if split_column not in self.feature_columns:
    #             print(f"Error: Column '{split_column}' not found in feature columns.")
    #             return None

    #         if not self.feature_columns[split_column]:
    #             print(f"Error: Column '{split_column}' is not marked as a feature.")
    #             return None
        
    #     try:
    #         split_data_dict = {}
    #         for split_column in split_columns:
    #             data_split = self.processed_data[self.processed_data[split_column] != 0] #Non zero values
    #             data_complement = self.processed_data[self.processed_data[split_column] == 0] #Zero Values

    #             X_with_intercept_split = np.column_stack([np.ones(len(data_split)), data_split])
    #             X_with_intercept_complement = np.column_stack([np.ones(len(data_complement)), data_complement])

    #             split_data_dict[split_column] = (X_with_intercept_split, X_with_intercept_complement)

    #         return split_data_dict

    #     except KeyError:
    #         print(f"KeyError: A column was not found")
    #         return None
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         return None

    # def split_data(self, split_column, drop_complement=True):
    #     if self.feature_columns[split_column]:
    #         np.column_stack([np.ones(len(X_extracurricular_yes)), self.processed_data])

# X_extracurricular_yes = updated_data.drop(['Performance Index', 'Extracurricular Activities_No'], axis=1)
# X_with_intercept_yes = np.column_stack([np.ones(len(X_extracurricular_yes)), X_extracurricular_yes])
# cols_yes = ['Intercept'] + X_extracurricular_yes.columns.tolist()

# X_extracurricular_no = updated_data.drop(['Performance Index', 'Extracurricular Activities_Yes'], axis=1)
# X_with_intercept_no = np.column_stack([np.ones(len(X_extracurricular_no)), X_extracurricular_no])
# cols_no = ['Intercept'] + X_extracurricular_no.columns.tolist()
        # """
        # Split the feature data based on a binary categorical column into two sets.
        
        # :param split_column: Name of the binary categorical column to split on (e.g., 'Extracurricular Activities')
        # :param drop_complement: Whether to drop the complementary column for each split (default: True)
        # :return: Dictionary containing feature matrices and column names for each category
        # """
        # # Verify the split column exists in categorical columns
        # if split_column not in self.categorical_columns:
        #     raise ValueError(f"{split_column} must be in categorical_columns and encoded")
        
        # # # Get the encoded column names for the split variable
        # split_cols = [col for col in self.feature_columns if col.startswith(f"{split_column}_")]
        # # if len(split_cols) != 2:
        # #     raise ValueError(f"{split_column} must be binary (exactly two categories after encoding)")
        
        # result = {}
        
        # # Process each category
        # for category_col in split_cols:
        #     # Get the rows where this category is True (1)
        #     category_mask = self.processed_data[category_col]
            
        #     # # Create feature matrix for this category
        #     X_category = self.processed_data[category_mask].copy()
            
        #     # # Drop the complement column if requested
        #     if drop_complement:
        #     #     # Find the other category column
        #         other_category = [col for col in split_cols if col != category_col][0]
        #         X_category = X_category.drop(columns=[other_category])
            
        #     # Add intercept column
        #     X_with_intercept = np.column_stack([
        #         np.ones(len(X_category)),
        #         X_category
        #     ])
            
        #     # Create column names list with intercept
        #     cols = ['Intercept'] + X_category.columns.tolist()
            
        #     # Store results for this category
        #     category_name = category_col.split('_')[-1]  # Extract category name (e.g., 'Yes' from 'Extracurricular Activities_Yes')
        #     result[category_name] = {
        #         'X': X_with_intercept,
        #         'columns': cols
        #     }
        
        # return result

    # def split_data(self, split_column):
    #     """
    #     Splits the data based on a dummy-coded column, creating separate DataFrames
    #     with 0s and 1s for each dummy column.

    #     Args:
    #         split_column: The name of the dummy-coded column to split on.

    #     Returns:
    #         A dictionary where keys are the dummy column names and values are tuples
    #         containing:
    #         - A NumPy array with intercept for rows where that dummy column is 1.
    #         - A NumPy array with intercept for rows where that dummy column is 0.
    #         Returns None if there is an error
    #     """

    #     if split_column not in self.feature_columns:
    #         print(f"Error: Column '{split_column}' not found in feature columns.")
    #         return None

    #     if not self.feature_columns[split_column]:
    #         print(f"Error: Column '{split_column}' is not marked as a feature.")
    #         return None

    #     try:
    #         dummy_cols = [col for col in self.processed_data.columns if col.startswith(split_column)]
    #         if not dummy_cols:
    #             print(f"Error: No dummy columns found starting with '{split_column}'.")
    #             return None
            
    #         split_data_dict = {}

    #         for dummy_col in dummy_cols:
    #             data_split = self.processed_data[self.processed_data[dummy_col] == 1]
    #             data_complement = self.processed_data[self.processed_data[dummy_col] == 0]

    #             X_with_intercept_split = np.column_stack([np.ones(len(data_split)), data_split])
    #             X_with_intercept_complement = np.column_stack([np.ones(len(data_complement)), data_complement])

    #             split_data_dict[dummy_col] = (X_with_intercept_split, X_with_intercept_complement)

    #         return split_data_dict

    #     except KeyError:
    #         print(f"KeyError: A column was not found")
    #         return None
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         return None

    # def split_data(self, split_column, drop_complement=True):
    #     """
    #     Split the data based on a binary categorical column into two sets.
        
    #     :param split_column: Name of the binary categorical column to split on (e.g., 'Extracurricular Activities')
    #     :param drop_complement: Whether to drop the complementary column for each split (default: True)
    #     :return: Dictionary containing feature matrices and column names for each category
    #     """
    #     # Verify the split column exists in categorical columns
    #     if split_column not in self.categorical_columns:
    #         raise ValueError(f"{split_column} must be in categorical_columns and encoded")
        
    #     # Get the encoded column names for the split variable
    #     split_cols = [col for col in self.feature_columns if col.startswith(f"{split_column}_")]
    #     if len(split_cols) != 2:
    #         raise ValueError(f"{split_column} must be binary (exactly two categories after encoding)")
        
    #     result = {}
        
    #     # Process each category
    #     for category_col in split_cols:
    #         # Get the rows where this category is True (1)
    #         category_mask = self.processed_data[category_col] == 1
            
    #         # Create feature matrix for this category
    #         X_category = self.processed_data[category_mask].copy()
            
    #         # Drop the complement column if requested
    #         if drop_complement:
    #             # Find the other category column
    #             other_category = [col for col in split_cols if col != category_col][0]
    #             X_category = X_category.drop(columns=[other_category])
            
    #         # Add intercept column
    #         X_with_intercept = np.column_stack([
    #             np.ones(len(X_category)),
    #             X_category
    #         ])
            
    #         # Create column names list with intercept
    #         cols = ['Intercept'] + X_category.columns.tolist()
            
    #         # Get corresponding target values
    #         y_category = self.target_data[category_mask]
            
    #         # Store results for this category
    #         category_name = category_col.split('_')[-1]  # Extract category name (e.g., 'Yes' from 'Extracurricular Activities_Yes')
    #         result[category_name] = {
    #             'X': X_with_intercept,
    #             'y': y_category,
    #             'columns': cols
    #         }
        
    #     return result
        # Initialize feature sets for cases
        # self._initialize_cases()

    # def _initialize_cases(self):
    #     """
    #     Create feature sets for each case based on categorical variables.
    #     """
    #     for column in self.categorical_columns:
    #         one_hot_columns = [col for col in self.processed_data.columns if col.startswith(f"{column}_")]
            
    #         for one_hot_col in one_hot_columns:
    #             X_case = self.processed_data.drop(columns=[col for col in one_hot_columns if col != one_hot_col], errors='ignore')
    #             X_with_intercept = np.column_stack([np.ones(len(X_case)), X_case])
    #             cols = ['Intercept'] + X_case.columns.tolist()
    #             self.feature_sets[one_hot_col] = {'X': X_with_intercept, 'cols': cols}
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    @staticmethod
    def plot_correlation_matrix(data):
        """
        Plot a correlation matrix of the data.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

    @staticmethod
    def plot_histograms(data, columns=None):
        """
        Plot histograms for specified columns or all numeric columns in the data.
        """
        if columns is None:
            columns = data.select_dtypes(include='number').columns
        
        data[columns].hist(figsize=(12, 10), bins=15, color='blue', alpha=0.7)
        plt.suptitle("Histograms of Numeric Features")
        plt.show()

    @staticmethod
    def plot_scatter_matrix(data, columns=None):
        """
        Plot a scatter matrix for the specified columns or all numeric columns.
        """
        if columns is None:
            columns = data.select_dtypes(include='number').columns
        
        sns.pairplot(data[columns], diag_kind='kde')
        plt.suptitle("Scatter Matrix", y=1.02)
        plt.show()

    @staticmethod
    def plot_box_plots(data, columns=None):
        """
        Plot box plots for specified columns or all numeric columns.
        """
        if columns is None:
            columns = data.select_dtypes(include='number').columns
        
        data_melted = data.melt(value_vars=columns)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="variable", y="value", data=data_melted)
        plt.title("Box Plots of Numeric Features")
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def plot_target_distribution(data, target_column):
        """
        Plot the distribution of the target variable.
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(data[target_column], kde=True, bins=15, color='green', alpha=0.7)
        plt.title(f"Distribution of Target Variable: {target_column}")
        plt.xlabel(target_column)
        plt.ylabel("Frequency")
        plt.show()


class LinearRegression:
    def __init__(self, features_yes, features_no, label, cols_yes, cols_no):
        self.features_yes = features_yes 
        self.features_no = features_no
        self.label = label
        self.cols_yes = cols_yes
        self.cols_no = cols_no

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