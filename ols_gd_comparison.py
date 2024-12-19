import kagglehub
import matplotlib.pyplot as plt
import numpy as np 
import os
import pandas as pd 
from scipy.linalg import pinv
import seaborn as sns 
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
import sys
import time
import traceback

class LinearRegression:
    def __init__(self, features_yes, features_no, label, cols_yes, cols_no):
        self.features_yes = features_yes 
        self.features_no = features_no
        self.label = label
        self.cols_yes = cols_yes
        self.cols_no = cols_no
        
        # Initialize comparison DataFrames
        self.comparison_yes = pd.DataFrame()
        self.comparison_yes.attrs['title'] = 'Comparison of Extracurricular - Yes Activities'
        
        self.comparison_no = pd.DataFrame()
        self.comparison_no.attrs['title'] = 'Comparison of Extracurricular - No Activities'
        
        # Add debug flag
        self.debug = True
    
    def _compute_parameters(self, features, key, cols, method='numpy'):
        start_time = time.time()
        
        # Choose computation method
        if method == 'numpy':
            beta_encoding = np.linalg.inv(features.T @ features) @ features.T @ self.label
        elif method == 'scipy':
            beta_encoding = pinv(features.T @ features) @ features.T @ self.label
            
        beta_series = pd.Series(data=beta_encoding, index=cols)
        elapsed_time = time.time() - start_time
        
        # Memory measurements
        beta_memory = sys.getsizeof(beta_encoding)
        series_memory = sys.getsizeof(beta_series)
        
        # Display results
        print(f"{method.capitalize()} Method:")
        print(f"Computed beta coefficients in {elapsed_time:.6f} seconds")
        print(f"Memory usage for parameters of '{key}' variable: {beta_memory} bytes")
        print(f"Memory usage for '{key}' variable Series: {series_memory} bytes")
        print(f"Beta coefficients {key}:")
        print(beta_series)
        print("\n")
        
        return beta_series
    
    def gradient_descent(self, features, key, cols, learning_rate=0.000001, epochs=100, precision=0.00001):
        """
        Perform gradient descent optimization with dimension checking
        """
        try:
            # Ensure features and cols match in dimension
            if self.debug:
                print(f"Features shape: {features.shape}")
                print(f"Number of columns: {len(cols)}")
                print(f"Columns: {cols}")
            
            # Split the data without the intercept column
            X = features[:, 1:] if features.shape[1] == len(cols) else features
            X_train, X_test, y_train, y_test = train_test_split(X, self.label, test_size=0.2, random_state=10)
            
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
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                X_train_augmented = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
                X_test_augmented = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))
            else:
                X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
                X_test_augmented = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
            
            # Verify dimensions
            if self.debug:
                print(f"X_train_augmented shape: {X_train_augmented.shape}")
                print(f"Number of coefficients to estimate: {len(cols)}")
            
            # Initialize parameters
            beta = np.zeros(len(cols))
            guesses = []
            losses = []
            
            # Gradient descent iterations
            for epoch in range(epochs):
                predictions = X_train_augmented @ beta
                residuals = predictions - y_train
                gradient = (2 / len(y_train)) * X_train_augmented.T @ residuals
                
                beta = beta - learning_rate * gradient
                
                guesses.append(beta.copy())
                loss = np.mean(residuals ** 2)
                losses.append(loss)
                
                if np.max(np.abs(gradient)) < precision:
                    break
            
            elapsed_time = time.time() - start_time
            beta_series = pd.Series(data=beta, index=cols)
            
            print(f"Gradient Descent ({key}):")
            print(f"Computed beta coefficients in {elapsed_time:.6f} seconds")
            print(f'Memory usage for Pandas Series: {sys.getsizeof(beta_series)} bytes')
            print(beta_series)
            print("\n")
            
            return beta_series, guesses, losses
            
        except Exception as e:
            print(f"Error in gradient_descent: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def train_sklearn_models(self, features, key, cols, ridge_alpha=1.0, lasso_alpha=1.0):
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
            raise
    
    def fit_numpy(self):
        """Compute parameters using NumPy method"""
        self.numpy_yes = self._compute_parameters(self.features_yes, 'yes', self.cols_yes, 'numpy')
        self.numpy_no = self._compute_parameters(self.features_no, 'no', self.cols_no, 'numpy')
        return self.numpy_yes, self.numpy_no
    
    def fit_scipy(self):
        """Compute parameters using SciPy method"""
        self.scipy_yes = self._compute_parameters(self.features_yes, 'yes', self.cols_yes, 'scipy')
        self.scipy_no = self._compute_parameters(self.features_no, 'no', self.cols_no, 'scipy')
        return self.scipy_yes, self.scipy_no
    
    def fit_statsmodels(self, features, key):
        """Compute parameters using statsmodels"""
        start_time = time.time()
        
        X_constant = sm.add_constant(features)
        model = sm.OLS(self.label, X_constant).fit()
        summary = model.summary()
        
        elapsed_time = time.time() - start_time
        
        print(f"Statsmodels Method ({key}):")
        print(f"Computed beta coefficients in {elapsed_time:.6f} seconds")
        print(f"Memory usage for: {sys.getsizeof(model)} bytes")
        print("\nModel Summary:")
        print(summary)
        print("\n")
        
        return model, summary
    
    def fit_all_statsmodels(self):
        """Compute statsmodels for both yes and no cases"""
        self.statsmodels_yes_model, self.statsmodels_yes_summary = self.fit_statsmodels(self.features_yes, 'yes')
        self.statsmodels_no_model, self.statsmodels_no_summary = self.fit_statsmodels(self.features_no, 'no')
        return (self.statsmodels_yes_model, self.statsmodels_yes_summary), (self.statsmodels_no_model, self.statsmodels_no_summary)
    
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

# Targeting data

def load_and_preprocess_data():
    """
    Load and preprocess the Student Performance dataset from Kaggle
    """
    # Download dataset
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")
    dataset_path = os.path.join(path, 'Student_Performance.csv')
    
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
    print(f"Number of features (Yes case): {len(cols_yes)}")
    print(f"Number of features (No case): {len(cols_no)}")
    print(f"Number of samples: {len(y)}")
    
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

def load_and_preprocess_data():
    """
    Load and preprocess the Student Performance dataset from Kaggle
    """
    # Download dataset
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("nikhil7280/student-performance-multiple-linear-regression")
    dataset_path = os.path.join(path, 'Student_Performance.csv')
    
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

def plot_data_overview(data_dict):
    """
    Create overview plots of the dataset using seaborn's default style
    """
    # Set the style using seaborn
    sns.set_theme()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance Index distribution
    sns.histplot(data=data_dict['processed_data'], x='Performance Index', ax=axes[0,0])
    axes[0,0].set_title('Distribution of Performance Index')
    
    # Performance by Extracurricular Activities
    sns.boxplot(data=data_dict['raw_data'], x='Extracurricular Activities', y='Performance Index', ax=axes[0,1])
    axes[0,1].set_title('Performance by Extracurricular Activities')
    
    # Correlation matrix for numerical features
    numerical_cols = data_dict['processed_data'].select_dtypes(include=[np.number]).columns
    correlation_matrix = data_dict['processed_data'][numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1,0])
    axes[1,0].set_title('Correlation Matrix')
    
    # Sample scatter plot
    sns.scatterplot(data=data_dict['raw_data'], x='Hours Studied', y='Performance Index', ax=axes[1,1])
    axes[1,1].set_title('Performance vs Hours Studied')
    
    plt.tight_layout()
    return fig

def run_analysis(save_results=True):
    """
    Run the complete analysis pipeline
    """
    try:
        # Load and preprocess data
        print("Starting data preprocessing...")
        data_dict = load_and_preprocess_data()
        
        # Create visualization
        print("\nCreating data visualizations...")
        fig = plot_data_overview(data_dict)
        if save_results:
            os.makedirs('results', exist_ok=True)
            fig.savefig('results/data_overview.png')
        plt.close()
        
        # Initialize the model
        print("\nInitializing model...")
        model = LinearRegression(
            data_dict['X_with_intercept_yes'],
            data_dict['X_with_intercept_no'],
            data_dict['y'],
            data_dict['cols_yes'],
            data_dict['cols_no']
        )
        model.debug = True  # Enable debug output
        
        # Fit all methods
        print("\nFitting all regression methods...")
        model.fit_all_methods()
        
        # Get comparisons
        comparison_yes, comparison_no = model.comparison_yes, model.comparison_no
        
        # Display results
        print("\nResults for students with extracurricular activities:")
        print(comparison_yes)
        print("\nResults for students without extracurricular activities:")
        print(comparison_no)
        
        # Save results
        if save_results:
            print("\nSaving results...")
            model.save_results()
            
        print("\nAnalysis complete!")
        return model, data_dict
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        model, data_dict = run_analysis(save_results=True)
        print("\nScript executed successfully!")
    except Exception as e:
        print(f"Script execution failed: {str(e)}")
        sys.exit(1)