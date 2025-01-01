# # # import pandas as pd
# # # import numpy as np

# # # class DataProcessor:
# # #     def __init__(self, data, feature_columns):
# # #         """Initializes the DataProcessor."""
# # #         self.data = data.copy()
# # #         self.feature_columns = feature_columns
# # #         self.processed_data = self.data[self.feature_columns].copy()

# # #     def create_yes_no_datasets(self, base_column_name):
# # #         """
# # #         Creates "Yes" and "No" DataFrames based on a base column name for dummy-coded variables.

# # #         Args:
# # #             base_column_name: The base name of the categorical variable (e.g., "Extracurricular Activities").

# # #         Returns:
# # #             A dictionary with keys 'Yes' and 'No', each containing a dictionary with 'X' (NumPy array with intercept) and 'columns' (list of column names).
# # #             Returns None if there is an error
# # #         """
# # #         try:
# # #             # Find all dummy columns related to the base name
# # #             dummy_cols = [col for col in self.processed_data.columns if col.startswith(base_column_name)]

# # #             if not dummy_cols:
# # #                 raise KeyError(f"No dummy columns found for base name: {base_column_name}")

# # #             # Create "Yes" DataFrame (where ANY of the dummy columns is 1)
# # #             yes_condition = self.processed_data[dummy_cols].any(axis=1)  #Check if any of the dummy columns are 1
# # #             yes_df = self.processed_data[yes_condition].copy()
# # #             X_with_intercept_yes = np.column_stack([np.ones(len(yes_df)), yes_df])
# # #             cols_yes = ['Intercept'] + yes_df.columns.tolist()

# # #             # Create "No" DataFrame (where ALL dummy columns are 0)
# # #             no_condition = self.processed_data[dummy_cols].all(axis=1) #Check if all dummy columns are 0
# # #             no_df = self.processed_data[~no_condition].copy()
# # #             X_with_intercept_no = np.column_stack([np.ones(len(no_df)), no_df])
# # #             cols_no = ['Intercept'] + no_df.columns.tolist()

# # #             return {'Yes': {'X': X_with_intercept_yes, 'columns': cols_yes},
# # #                     'No': {'X': X_with_intercept_no, 'columns': cols_no}}

# # #         except KeyError as e:
# # #             print(f"KeyError: {e}")
# # #             return None
# # #         except Exception as e:
# # #             print(f"An unexpected error occurred: {e}")
# # #             return None

# # # # Example Usage:
# # # data = {'Performance Index': [1, 2, 3, 4, 5, 6],
# # #         'Extracurricular Activities_A': [1, 0, 0, 0, 1, 0],
# # #         'Extracurricular Activities_B': [0, 1, 0, 1, 0, 0],
# # #         'Extracurricular Activities_C': [0, 0, 1, 0, 0, 1],
# # #         'Other_Feature': [4, 5, 6, 7, 8, 9],
# # #         'Another_Feature': [10, 11, 12, 13, 14, 15]}

# # # feature_columns = ['Extracurricular Activities_A', 'Extracurricular Activities_B', 'Extracurricular Activities_C', 'Other_Feature', 'Another_Feature']
# # # df = pd.DataFrame(data)
# # # processor = DataProcessor(df, feature_columns)

# # # split_data = processor.create_yes_no_datasets('Extracurricular Activities')

# # # if split_data:
# # #     X_yes = split_data['Yes']['X']
# # #     cols_yes = split_data['Yes']['columns']

# # #     X_no = split_data['No']['X']
# # #     cols_no = split_data['No']['columns']

# # #     print("Yes Data:\n", X_yes)
# # #     print("Yes Columns:\n", cols_yes)
# # #     print("No Data:\n", X_no)
# # #     print("No Columns:\n", cols_no)
# # # else:
# # #     print("An error occurred during data splitting.")

# # # #Example with incorrect base name
# # # split_data_incorrect_base_name = processor.create_yes_no_datasets("Not_a_base_name")

# # import pandas as pd
# # import numpy as np

# # class DataProcessor:
# #     def __init__(self, data, target_column, categorical_columns=None):
# #         """Initializes the DataProcessor."""
# #         self.data = data.copy()
# #         self.target_column = target_column
# #         self.categorical_columns = categorical_columns if categorical_columns else []
# #         self._preprocess_data()

# #     def _preprocess_data(self):
# #         """
# #         Perform preprocessing, such as removing nulls, duplicates, and encoding categorical variables.
# #         """
# #         # Drop nulls and duplicates
# #         self.data.dropna(inplace=True)
# #         self.data.drop_duplicates(inplace=True)
        
# #         # Copy raw data to processed data
# #         self.processed_data = self.data.copy()
        
# #         # Extract target variable first
# #         self.target_data = self.processed_data.pop(self.target_column)
        
# #         # Perform dummy coding if categorical columns exist
# #         if self.categorical_columns:
# #             self._dummy_coding()
        
# #         # Update feature columns
# #         self.feature_columns = self.processed_data.columns.tolist()

# #     def _dummy_coding(self):
# #         """
# #         Perform one-hot encoding on categorical columns if specified.
# #         Updates processed_data and feature_columns.
# #         """
# #         if not len(self.data):
# #             return
            
# #         if self.categorical_columns:
# #             self.processed_data = pd.get_dummies(
# #                 self.data.drop(columns=[self.target_column]), 
# #                 columns=self.categorical_columns, 
# #                 dtype=int
# #             )
# #         else:
# #             self.processed_data = self.data.drop(columns=[self.target_column]).copy()
            
# #         self.feature_columns = self.processed_data.columns.tolist()
# #         self.target_data = self.data[self.target_column]
    
# #     def split_by_dummy_group(self, base_column_name):
# #         """Splits data by existing dummy columns or creates them if needed."""
# #         try:
# #             dummy_cols = [col for col in self.processed_data.columns if col.startswith(base_column_name)]
# #             if not dummy_cols:  # If no pre-existing dummy columns are found
# #                 if base_column_name in self.processed_data.columns:
# #                     #Create dummy columns
# #                     dummies = pd.get_dummies(self.processed_data[base_column_name], prefix=base_column_name, dtype=int)
# #                     self.processed_data = pd.concat([self.processed_data, dummies], axis=1)
# #                     dummy_cols = [col for col in self.processed_data.columns if col.startswith(base_column_name)]
# #                 else:
# #                     raise KeyError(f"Column '{base_column_name}' not found in processed data.")
# #             split_dataframes = {}
# #             for dummy_col in dummy_cols:
# #                 df = self.processed_data[self.processed_data[dummy_col] == 1].copy()
# #                 split_dataframes[dummy_col] = df
# #             return split_dataframes

# #         except KeyError as e:
# #             print(f"KeyError: {e}")
# #             return None
# #         except Exception as e:
# #             print(f"An unexpected error occurred: {e}")
# #             return None

# #     def create_yes_no_datasets_from_split(self, split_dataframes, base_column_name):
# #         try:
# #             yes_key = [key for key in split_dataframes if key.endswith("_Yes") or key.endswith("_yes")][0]
# #             no_key = [key for key in split_dataframes if key.endswith("_No") or key.endswith("_no")][0]
# #             yes_df = split_dataframes[yes_key]
# #             no_df = split_dataframes[no_key]
# #             return {'Yes': yes_df, 'No': no_df}

# #         except IndexError as e:
# #             print("Could not find the yes or no key")
# #             return None
# #         except Exception as e:
# #             print(f"An unexpected error occurred: {e}")
# #             return None


# # # Example Usage (with standard one-hot encoding):
# # data2 = {'Performance Index': [1, 2, 3, 4, 5, 6],
# #         'Extracurricular Activities': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
# #         'Other_Feature': [4, 5, 6, 7, 8, 9],
# #         'Another_Feature': [10, 11, 12, 13, 14, 15]}
# # feature_columns2 = ['Extracurricular Activities', 'Other_Feature', 'Another_Feature']
# # df2 = pd.DataFrame(data2)
# # processor2 = DataProcessor(df2, "Performance Index", ['Extracurricular Activities'])

# # split_dfs2 = processor2.split_by_dummy_group('Extracurricular Activities')

# # if split_dfs2:
# #     print("\nSplit DataFrames (from one-hot encoding):")
# #     for name, df in split_dfs2.items():
# #         print(f"\n{name}:\n{df}")
# #     yes_no_dfs2 = processor2.create_yes_no_datasets_from_split(split_dfs2, "Extracurricular Activities")
# #     if yes_no_dfs2:
# #         print("\nYes DataFrame:\n", yes_no_dfs2['Yes'])
# #         print("\nNo DataFrame:\n", yes_no_dfs2['No'])
# #     else:
# #         print("Could not create the yes no dataframes")
# # else:
# #     print("Could not create the split dataframes")


# # # Example Usage (with pre-existing dummy columns):
# # data = {'Performance Index': [1, 2, 3, 4, 5, 6],
# #         'extracurriculars_yes': [1, 0, 0, 0, 1, 0],
# #         'extracurriculars_no': [0, 1, 0, 1, 0, 0],
# #         'Other_Feature': [4, 5, 6, 7, 8, 9],
# #         'Another_Feature': [10, 11, 12, 13, 14, 15]}

# # feature_columns = ['extracurriculars_yes', 'extracurriculars_no', 'Other_Feature', 'Another_Feature']
# # df = pd.DataFrame(data)
# # processor = DataProcessor(df, "Performance Index",)

# # split_dfs = processor.split_by_dummy_group('extracurriculars')

# # if split_dfs:
# #     print("\nSplit DataFrames (from pre-existing dummy columns):")
# #     for name, df in split_dfs.items():
# #         print(f"\n{name}:\n{df}")
# #     yes_no_dfs = processor.create_yes_no_datasets_from_split(split_dfs, "extracurriculars")

# #     if yes_no_dfs:
# #         print("\nYes DataFrame:\n", yes_no_dfs['Yes'])
# #         print("\nNo DataFrame:\n", yes_no_dfs['No'])
# #     else:
# #         print("Could not create the yes no dataframes")
# # else:
# #     print("Could not create the split dataframes")

# import pandas as pd
# import numpy as np

# class DataProcessor:
#     def __init__(self, data, target_column, categorical_columns=None):
#         """Initializes the DataProcessor."""
#         self.data = data.copy()
#         self.target_column = target_column
#         self.categorical_columns = categorical_columns if categorical_columns else []
#         self._preprocess_data()

#     def _preprocess_data(self):
#         """
#         Perform preprocessing, such as removing nulls, duplicates, and encoding categorical variables.
#         """
#         # Drop nulls and duplicates
#         self.data.dropna(inplace=True)
#         self.data.drop_duplicates(inplace=True)
        
#         # Copy raw data to processed data
#         self.processed_data = self.data.copy()
        
#         # Extract target variable first
#         self.target_data = self.processed_data.pop(self.target_column)
        
#         # Perform dummy coding if categorical columns exist
#         if self.categorical_columns:
#             self._dummy_coding()
        
#         # Update feature columns
#         self.feature_columns = self.processed_data.columns.tolist()

#     def _dummy_coding(self):
#         """Performs one-hot encoding."""
#         if self.categorical_columns:
#             self.processed_data = pd.get_dummies(
#                 self.data.drop(columns=[self.target_column]),
#                 columns=self.categorical_columns,
#                 dtype=int,
#                 prefix_sep='_' #Important for the split method to work
#             )
#         else:
#             self.processed_data = self.data.drop(columns=[self.target_column]).copy()
#         self.feature_columns = self.processed_data.columns.tolist()
#         self.target_data = self.data[self.target_column]
    
#     def create_yes_no_datasets(self, base_column_name):
#         """
#         Creates "Yes" and "No" DataFrames after one-hot encoding.

#         Args:
#             base_column_name: The base name of the categorical variable (e.g., "Extracurricular Activities").

#         Returns:
#             A dictionary with keys 'Yes' and 'No', each containing a DataFrame.
#             Returns None if there is an error
#         """
#         try:
#             # Find all dummy columns related to the base name
#             dummy_cols = [col for col in self.processed_data.columns if col.startswith(base_column_name + "_")]
#             if not dummy_cols:
#                 raise KeyError(f"No dummy columns found for base name: {base_column_name}")
            
#             yes_key = [key for key in dummy_cols if key.endswith("_Yes") or key.endswith("_yes")][0]
#             no_key = [key for key in dummy_cols if key.endswith("_No") or key.endswith("_no")][0]

#             yes_df = self.processed_data[self.processed_data[yes_key] == 1].copy()
#             no_df = self.processed_data[self.processed_data[no_key] == 1].copy()

#             return {'Yes': yes_df, 'No': no_df}

#         except KeyError as e:
#             print(f"KeyError: {e}")
#             return None
#         except IndexError as e:
#             print(f"IndexError: {e}. Ensure that the categories Yes and No exist")
#             return None
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")
#             return None

# # Example Usage:
# data = {'Performance Index': [1, 2, 3, 4, 5, 6],
#         'Extracurricular Activities': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
#         'Other_Feature': [4, 5, 6, 7, 8, 9],
#         'Another_Feature': [10, 11, 12, 13, 14, 15]}

# feature_columns = ['Extracurricular Activities', 'Other_Feature', 'Another_Feature']
# df = pd.DataFrame(data)
# processor = DataProcessor(df, "Performance Index", ['Extracurricular Activities'])

# split_data = processor.create_yes_no_datasets('Extracurricular Activities')

# if split_data:
#     yes_df = split_data['Yes']
#     no_df = split_data['No']

#     print("Yes DataFrame:\n", yes_df)
#     print("No DataFrame:\n", no_df)
# else:
#     print("An error occurred during DataFrame creation.")

# #Example with incorrect base name
# split_data_incorrect_base_name = processor.create_yes_no_datasets("Not_a_base_name")

# #Example with missing yes or no category
# data_missing_category = {'Performance Index': [1, 2, 3, 4, 5, 6],
#         'Extracurricular Activities': ['Maybe', 'Maybe', 'Maybe', 'Maybe', 'Maybe', 'Maybe'],
#         'Other_Feature': [4, 5, 6, 7, 8, 9],
#         'Another_Feature': [10, 11, 12, 13, 14, 15]}
# processor_missing_category = DataProcessor(data_missing_category, "Performance Index", ['Extracurricular Activities'])
# split_data_missing_category = processor_missing_category.create_yes_no_datasets("Extracurricular Activities")

import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, data, target_column, categorical_columns=None):
        """Initializes the DataProcessor."""
        self.data = data.copy()
        self.target_column = target_column
        self.categorical_columns = categorical_columns if categorical_columns else []
        self._preprocess_data()

    def _preprocess_data(self):
        """
        Perform preprocessing, such as removing nulls, duplicates, and encoding categorical variables.
        """
        # Drop nulls and duplicates
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        
        # Copy raw data to processed data
        self.processed_data = self.data.copy()
        
        # Extract target variable first
        self.target_data = self.processed_data.pop(self.target_column)
        
        # Perform dummy coding if categorical columns exist
        if self.categorical_columns:
            self._dummy_coding()
        
        # Update feature columns
        self.feature_columns = self.processed_data.columns.tolist()

    def _dummy_coding(self):
        """Performs one-hot encoding."""
        if self.categorical_columns:
            self.processed_data = pd.get_dummies(
                self.data.drop(columns=[self.target_column]),
                columns=self.categorical_columns,
                dtype=int,
                prefix_sep='_'
            )
        else:
            self.processed_data = self.data.drop(columns=[self.target_column]).copy()
        self.feature_columns = self.processed_data.columns.tolist()
        self.target_data = self.data[self.target_column]
    
    # def create_yes_no_datasets(self, base_column_name):
    #     """
    #     Creates "Yes" and "No" DataFrames after one-hot encoding.

    #     Args:
    #         base_column_name: The base name of the categorical variable (e.g., "Extracurricular Activities").

    #     Returns:
    #         A dictionary with keys 'Yes' and 'No', each containing a DataFrame.
    #         Returns None if there is an error
    #     """
    #     try:
    #         # Find all dummy columns related to the base name
    #         dummy_cols = [col for col in self.processed_data.columns if col.startswith(base_column_name + "_")]
    #         if not dummy_cols:
    #             raise KeyError(f"No dummy columns found for base name: {base_column_name}")
            
    #         yes_key = [key for key in dummy_cols if key.endswith("_Yes") or key.endswith("_yes")][0]
    #         no_key = [key for key in dummy_cols if key.endswith("_No") or key.endswith("_no")][0]

    #         yes_df = self.processed_data[self.processed_data[yes_key] == 1].copy()
    #         no_df = self.processed_data[self.processed_data[no_key] == 1].copy()

    #         return {'Yes': yes_df, 'No': no_df}

    #     except KeyError as e:
    #         print(f"KeyError: {e}")
    #         return None
    #     except IndexError as e:
    #         print(f"IndexError: {e}. Ensure that the categories Yes and No exist")
    #         return None
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         return None

    # def create_yes_no_datasets(self, base_column_name):
    #     """Creates "Yes" and "No" DataFrames."""
    #     try:
    #         dummy_cols = [col for col in self.processed_data.columns if col.startswith(base_column_name)]

    #         if not dummy_cols:
    #             raise KeyError(f"No dummy columns found for base name: {base_column_name}")

    #         yes_dummy_col = [col for col in dummy_cols if col.endswith("_Yes") or col.endswith("_yes")][0]
    #         no_dummy_col = [col for col in dummy_cols if col.endswith("_No") or col.endswith("_no")][0]

    #         if not yes_dummy_col or not no_dummy_col:
    #             raise IndexError("Could not find both Yes and No dummy columns")

    #         # Correct way to create the DataFrames
    #         yes_df = self.processed_data[self.processed_data[yes_dummy_col] == 1].copy()
    #         no_df = self.processed_data[self.processed_data[no_dummy_col] == 1].copy()

    #         return {'Yes': yes_df, 'No': no_df}

    #     except KeyError as e:
    #         print(f"KeyError: {e}")
    #         return None
    #     except IndexError as e:
    #         print(f"IndexError: {e}. Ensure that the categories Yes and No exist")
    #         return None
    #     except Exception as e:
    #         print(f"An unexpected error occurred: {e}")
    #         return None


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

# Example Usage:
data = {'Performance Index': [1, 2, 3, 4, 5, 6, None],
        'Extracurricular Activities': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes'],
        'Other_Feature': [4, 5, 6, 7, 8, 9, None],
        'Another_Feature': [10, 11, 12, 13, 14, 15, 16]}

feature_columns = ['Extracurricular Activities', 'Other_Feature', 'Another_Feature']
df = pd.DataFrame(data)
processor = DataProcessor(df, "Performance Index", ['Extracurricular Activities'])

split_data = processor.create_yes_no_datasets('Extracurricular Activities')

if split_data:
    yes_df = split_data['Yes']
    no_df = split_data['No']

    print("Yes DataFrame:\n", yes_df)
    print("No DataFrame:\n", no_df)
else:
    print("An error occurred during DataFrame creation.")