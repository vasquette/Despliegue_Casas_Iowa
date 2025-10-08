import streamlit as st
import pandas as pd
import joblib
import numpy as np
import scipy.sparse

# --- Cargar Modelos y Preprocesadores ---
try:
    loaded_scaler = joblib.load('minmax_scaler.pkl')
    exterqual_label_encoder = joblib.load('ExterQual_label_encoder.pkl')
    extercond_label_encoder = joblib.load('ExterCond_label_encoder.pkl')
    bsmtqual_label_encoder = joblib.load('BsmtQual_label_encoder.pkl')
    bsmtcond_label_encoder = joblib.load('BsmtCond_label_encoder.pkl')
    heatingqc_label_encoder = joblib.load('HeatingQC_label_encoder.pkl')
    kitchenqual_label_encoder = joblib.load('KitchenQual_label_encoder.pkl')
    functional_label_encoder = joblib.load('Functional_label_encoder.pkl')
    garagefinish_label_encoder = joblib.load('GarageFinish_label_encoder.pkl')
    onehot_encoder = joblib.load('onehot_encoder.pkl')
    pca_model = joblib.load('pca_model.pkl')
    loaded_model = joblib.load('best_stacking_regressor_model.pkl')

    st.success("Modelos y preprocesadores cargados correctamente.")
    models_loaded = True
except FileNotFoundError as e:
    st.error(f"Error loading file: {e}. Please ensure all .pkl files are in the correct directory.")
    models_loaded = False
except Exception as e:
    st.error(f"An error occurred while loading models or preprocessors: {e}")
    models_loaded = False

# --- Define expected features after preprocessing (should match PCA model's feature_names_in_) ---
# This list should be generated during training and saved
if models_loaded and hasattr(pca_model, 'feature_names_in_'):
    EXPECTED_FEATURES_AFTER_PREPROCESSING = pca_model.feature_names_in_
else:
    EXPECTED_FEATURES_AFTER_PREPROCESSING = [] # Fallback

# --- Function to Generate Default Data (80 variables) and Input Widgets ---
def get_input_data_from_widgets():
    """Generates input widgets for each variable and returns a DataFrame."""
    st.sidebar.header('Enter House Characteristics')

    input_data = {}
    widget_keys = set()

    # List of all potential original columns based on the dataset
    all_original_cols = [
        'Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
        'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
        'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
        'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
        'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
        'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
        'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
        'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
        'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence',
        'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
        'SaleCondition', 'SalePrice' # Including SalePrice potentially, but it's target
    ]
    # Exclude 'SalePrice' and 'Id' from widget generation
    cols_for_widgets = [col for col in all_original_cols if col not in ['Id', 'SalePrice']]

    # Define common options for categorical variables (can be expanded based on training data analysis)
    categorical_options = {
        'MSZoning': ['RL', 'RM', 'FV', 'RH', 'A', 'C (all)', 'I'], # Include 'C (all)' for completeness, though handled in preprocess
        'Street': ['Pave', 'Grvl'],
        'Alley': [np.nan, 'Grvl', 'Pave'],
        'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
        'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],
        'Utilities': ['AllPub', 'NoSeWa'],
        'LotConfig': ['Inside', 'Frontage', 'Corner', 'CulDSac', 'FR2', 'FR3'],
        'LandSlope': ['Gtl', 'Mod', 'Sev'],
        'Neighborhood': ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber', 'IDOTC', 'ClearCr', 'StoneBr', 'SWISU', 'Blmngtn', 'MeadowV', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'],
        'Condition1': ['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe'],
        'Condition2': ['Norm', 'Artery', 'Feedr', 'RRNn', 'RRAn', 'RRAe', 'PosN', 'PosA'],
        'BldgType': ['1Fam', '2FmCon', 'Duplex', 'TwnhsE', 'Twnhs'],
        'HouseStyle': ['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'],
        'RoofStyle': ['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed'],
        'RoofMatl': ['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Roll', 'ClyTile'],
        'Exterior1st': ['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'CBlock', 'BrkComm', 'AsphShn', 'VinylSd', 'ImStucc', 'Stone', 'Other'],
        'Exterior2nd': ['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'BrkFace', 'Wd Sdng', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'CBlock', 'BrkComm', 'AsphShn', 'VinylSd', 'ImStucc', 'Stone', 'Other'],
        'MasVnrType': ['None', 'BrkFace', 'Stone', 'BrkCmn', np.nan],
        'Foundation': ['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'],
        'BsmtQual': ['Gd', 'TA', 'Ex', 'Fa', np.nan],
        'BsmtCond': ['TA', 'Gd', 'Fa', 'Po', np.nan],
        'BsmtExposure': ['No', 'Gd', 'Mn', 'Av', np.nan],
        'BsmtFinType1': ['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ', np.nan],
        'BsmtFinType2': ['Unf', 'Rec', 'LwQ', 'GLQ', 'ALQ', 'BLQ', np.nan],
        'Heating': ['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'],
        'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
        'CentralAir': ['Y', 'N'],
        'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', np.nan],
        'KitchenQual': ['Gd', 'TA', 'Ex', 'Fa'],
        'Functional': ['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev'],
        'FireplaceQu': [np.nan, 'Gd', 'TA', 'Fa', 'Ex', 'Po'],
        'GarageType': ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', np.nan, 'Basment', '2Types'],
        'GarageFinish': ['RFn', 'Unf', 'Fin', np.nan], # Assuming these are the correct options
        'PavedDrive': ['Y', 'N', 'P'], # Assuming these are the correct options
        'PoolQC': [np.nan, 'Ex', 'Gd', 'TA', 'Fa'],
        'Fence': [np.nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'],
        'MiscFeature': [np.nan, 'Shed', 'Gar2', 'Othr', 'TenC'],
        'SaleType': ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'Con', 'Lwt', 'ConLw', np.nan], # Assuming these are the correct options
        'SaleCondition': ['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial', np.nan] # Assuming these are the correct options
    }

    # Define default values for numerical columns (can be refined)
    numerical_defaults = {
        'MSSubClass': 20, 'LotFrontage': 70.0, 'LotArea': 10500, 'OverallQual': 5,
        'OverallCond': 5, 'YearBuilt': 1980, 'YearRemodAdd': 1980, 'MasVnrArea': 0.0,
        'BsmtFinSF1': 500, 'BsmtFinSF2': 0, 'BsmtUnfSF': 500, 'TotalBsmtSF': 1000,
        '1stFlrSF': 1000, '2ndFlrSF': 0, 'LowQualFinSF': 0, 'GrLivArea': 1000,
        'BsmtFullBath': 0, 'BsmtHalfBath': 0, 'FullBath': 1, 'HalfBath': 0,
        'BedroomAbvGr': 3, 'KitchenAbvGr': 1, 'TotRmsAbvGrd': 5, 'Fireplaces': 0,
        'GarageYrBlt': 1980.0, 'GarageCars': 1, 'GarageArea': 300,
        'WoodDeckSF': 0, 'OpenPorchSF': 0, 'EnclosedPorch': 0, '3SsnPorch': 0,
        'ScreenPorch': 0, 'PoolArea': 0, 'MiscVal': 0, 'MoSold': 6, 'YrSold': 2008
    }


    # Generate widgets based on defined options and defaults
    for col in cols_for_widgets:
        widget_key = col
        widget_keys.add(widget_key)

        if col in categorical_options:
            options = categorical_options[col]
            # --- Improved Default Value Selection Logic ---
            default_value = None
            common_defaults = {
                'MSZoning': 'RL',
                'Street': 'Pave',
                'Alley': np.nan,
                'LotShape': 'Reg',
                'LandContour': 'Lvl',
                'Utilities': 'AllPub',
                'LotConfig': 'Inside',
                'LandSlope': 'Gtl',
                'Neighborhood': 'NAmes', # Or a more frequent one if known
                'Condition1': 'Norm',
                'Condition2': 'Norm',
                'BldgType': '1Fam',
                'HouseStyle': '1Story', # Or 2Story depending on commonality
                'RoofStyle': 'Gable',
                'RoofMatl': 'CompShg',
                'Exterior1st': 'VinylSd', # Or HdBoard/MetalSd
                'Exterior2nd': 'VinylSd', # Or HdBoard/MetalSd
                'MasVnrType': 'None', # Common for no veneer
                'Foundation': 'PConc', # Or CBlock
                'BsmtQual': 'TA', # Common quality
                'BsmtCond': 'TA', # Common condition
                'BsmtExposure': 'No', # Common exposure
                'BsmtFinType1': 'Unf', # Or GLQ
                'BsmtFinType2': 'Unf', # Most common
                'Heating': 'GasA',
                'HeatingQC': 'Ex', # Most common high quality
                'CentralAir': 'Y',
                'Electrical': 'SBrkr', # Most modern
                'KitchenQual': 'TA', # Common quality
                'Functional': 'Typ', # Typical functionality
                'FireplaceQu': np.nan, # Common if no fireplace
                'GarageType': 'Attchd', # Common
                'GarageFinish': 'Unf', # Or RFn
                'PavedDrive': 'Y', # Most common
                'PoolQC': np.nan, # Common if no pool
                'Fence': np.nan, # Common if no fence
                'MiscFeature': np.nan, # Common if no misc feature
                'SaleType': 'WD', # Warranty Deed
                'SaleCondition': 'Normal' # Normal sale
            }

            if col in common_defaults:
                 # Check if the intended default value is actually in the options for this column
                 if common_defaults[col] in options:
                     default_value = common_defaults[col]
                 elif pd.isna(common_defaults[col]) and np.nan in options:
                      default_value = np.nan
                 else:
                      # Fallback if the specific common default isn't available
                      if np.nan in options:
                           default_value = np.nan # Use NaN if available
                      else:
                           default_value = options[0] # Fallback to the first option
            else:
                # Fallback for columns not in common_defaults
                if np.nan in options:
                     default_value = np.nan # Use NaN if available
                else:
                     default_value = options[0] # Fallback to the first option


            # Find the index of the determined default value
            try:
                 # Ensure default_value is in options before trying to find its index
                 if default_value not in options and not pd.isna(default_value) and np.nan in options:
                      default_value = np.nan # If specific default not in options, try NaN if available
                 elif default_value not in options and not pd.isna(default_value):
                       default_value = options[0] # If specific default not in options and NaN not available, use first option

                 default_index = options.index(default_value)
            except ValueError:
                 # This fallback should ideally not be needed with the check above, but for safety
                 default_index = 0 # Fallback to index 0

            # Use the calculated default_index here
            input_data[widget_key] = st.sidebar.selectbox(col, options=options, index=default_index, key=widget_key)

        elif col in numerical_defaults:
            default_value = numerical_defaults[col]
            if isinstance(default_value, int):
                 input_data[widget_key] = st.sidebar.number_input(col, min_value=0, value=default_value, step=1, key=widget_key)
            else: # Assume float
                 input_data[widget_key] = st.sidebar.number_input(col, min_value=0.0, value=default_value, step=10.0, key=widget_key)
        else:
            # Fallback for any other columns (e.g., boolean or less common types)
            input_data[widget_key] = st.sidebar.text_input(col, key=widget_key)


    # Id is not a Streamlit widget and does not require a key.
    input_data['Id'] = 0 # A placeholder ID

    # Convert the dictionary to DataFrame
    return pd.DataFrame([input_data])


# --- Function to Preprocess Input Data ---
def preprocess_input(input_df):
    """Applies the same preprocessing steps used in training."""
    df_processed = input_df.copy()

    # Ensure all expected original columns exist in the input DataFrame
    # This list should cover all columns used in subsequent preprocessing steps
    expected_original_cols_for_processing = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
        'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
        'MiscVal', 'MoSold', 'YrSold', 'LotFrontage', 'GarageYrBlt', 'BsmtFinSF2', 'BsmtUnfSF',
        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
        # Categorical columns that might be OHE or Label Encoded
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
        'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
        'SaleType', 'SaleCondition'
    ]


    # Add missing columns with NaN
    for col in expected_original_cols_for_processing:
        if col not in df_processed.columns:
            df_processed[col] = np.nan

    # Select only the expected original columns needed for preprocessing
    df_processed = df_processed[expected_original_cols_for_processing].copy()


    # 1. Apply LabelEncoder (handle unseen labels)
    label_encode_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'Functional', 'GarageFinish']
    label_encoders = {
        'ExterQual': exterqual_label_encoder,
        'ExterCond': extercond_label_encoder,
        'BsmtQual': bsmtqual_label_encoder,
        'BsmtCond': bsmtcond_label_encoder,
        'HeatingQC': heatingqc_label_encoder,
        'KitchenQual': kitchenqual_label_encoder,
        'Functional': functional_label_encoder,
        'GarageFinish': garagefinish_label_encoder
    }

    for col in label_encode_cols:
        if col in df_processed.columns: # Check if column exists before processing
             encoder = label_encoders[col]
             try:
                 # Ensure the column is of object or category dtype before attempting cat.codes
                 if not pd.api.types.is_object_dtype(df_processed[col]) and not pd.api.types.is_categorical_dtype(df_processed[col]):
                      # Convert to string, handle NaN representations like 'nan'
                     df_processed[col] = df_processed[col].astype(str).replace('nan', np.nan)


                 # Handle unseen labels: convert to CategoricalDtype with known categories
                 # Fill NaNs before converting to categorical to avoid them being treated as a new category
                 # Ensure 'Missing' is in categories if filling NaNs with 'Missing'
                 known_categories = list(encoder.classes_)
                 if 'Missing' not in known_categories:
                      known_categories.append('Missing')

                 df_processed[col] = df_processed[col].fillna('Missing').astype(pd.CategoricalDtype(categories=known_categories))
                 df_processed[col] = df_processed[col].cat.codes
                 df_processed[col] = df_processed[col].replace(-1, np.nan) # Convert -1 from cat.codes (unseen) to NaN
                 df_processed[col] = df_processed[col].fillna(0) # Fill NaNs resulting from unseen/missing

             except Exception as e:
                 st.error(f"Could not apply LabelEncoder to column '{col}': {e}") # Changed from warning to error for clarity on critical failure
                 # Fallback: convert to numeric with errors='coerce' and fill NaNs
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(0)


    # Convert LabelEncoder columns (now numeric) to numeric type if they are not already
    for col in label_encode_cols:
        if col in df_processed.columns:
             if not pd.api.types.is_numeric_dtype(df_processed[col]):
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(0)


    # 3. Apply One-Hot Encoding (Dynamically determine columns from encoder and ensure all expected columns are generated)
    ohe_cols_to_transform = []
    unique_ohe_original_cols_expected = []
    full_expected_ohe_feature_names = [] # The full list of OHE column names expected by the PCA

    if models_loaded and hasattr(onehot_encoder, 'get_feature_names_out'):
        try:
            full_expected_ohe_feature_names = list(onehot_encoder.get_feature_names_out())

            # Extract original column names from encoder's transformed feature names
            for transformed_name in full_expected_ohe_feature_names:
                original_col = transformed_name.split('__')[0]
                if original_col not in unique_ohe_original_cols_expected:
                    unique_ohe_original_cols_expected.append(original_col)

            # Filter the input DataFrame to keep *only* these identified original columns and ensure they are in the correct order.
            ohe_cols_to_transform = [col for col in unique_ohe_original_cols_expected if col in df_processed.columns]

            # Verify that all expected OHE original columns are present in the input DataFrame
            if len(ohe_cols_to_transform) != len(unique_ohe_original_cols_expected):
                missing_cols = set(unique_ohe_original_cols_expected) - set(ohe_cols_to_transform)
                if missing_cols: # Only show error if there are missing columns
                    # Removed warning print for cleaner UI
                    # st.error(f"Warning: Input DataFrame is missing {len(missing_cols)} columns expected by the One-Hot Encoder: {missing_cols}. These will be skipped from OHE input but added with zeros later.")
                    pass # Suppress warning
                ohe_cols_to_transform = [col for col in unique_ohe_original_cols_expected if col in df_processed.columns] # Re-filter to be safe


        except Exception as e:
            st.error(f"Could not determine expected OHE columns from the encoder using get_feature_names_out: {e}. Cannot proceed with OHE.")
            return None # Indicate failure

    else:
        st.error("Could not determine expected OHE columns from the encoder (get_feature_names_out not available). Cannot proceed with OHE.")
        return None # Indicate failure


    if ohe_cols_to_transform:
        # Ensure the columns are in the correct order for transformation
        df_processed_ohe_subset = df_processed[ohe_cols_to_transform].copy()

        try:
            # Ensure handle_unknown='ignore' is set for the onehot_encoder
            if hasattr(onehot_encoder, 'handle_unknown'):
                 onehot_encoder.handle_unknown = 'ignore'
            else:
                st.warning("OneHotEncoder does not support handle_unknown='ignore'. Manual handling of unseen categories is crucial.")


            # Convert columns to string type to avoid issues with mixed types before OHE
            for col in ohe_cols_to_transform:
                 if col in df_processed_ohe_subset.columns:
                     # Convert any non-string/non-NaN values to string, handle NaNs
                     df_processed_ohe_subset[col] = df_processed_ohe_subset[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)
                     # Fill remaining NaNs with a placeholder string if 'nan' is not an expected category
                     if df_processed_ohe_subset[col].isnull().any():
                         try:
                             # Find the index of this original column within the expected OHE columns list
                             col_original_index = unique_ohe_original_cols_expected.index(col)
                             # Check if 'nan' is among the categories for this original column in the encoder
                             if col_original_index < len(onehot_encoder.categories_):
                                 encoder_cats_for_col = [str(cat) for cat in onehot_encoder.categories_[col_original_index] if pd.notna(cat)]
                                 if 'nan' not in encoder_cats_for_col:
                                      df_processed_ohe_subset[col] = df_processed_ohe_subset[col].fillna('Missing') # Fill with a placeholder if 'nan' not expected
                                 else:
                                      df_processed_ohe_subset[col] = df_processed_ohe_subset[col].fillna('nan') # Fill with 'nan' string if encoder expects it
                             else:
                                  df_processed_ohe_subset[col] = df_processed_ohe_subset[col].fillna('Missing') # Default to placeholder if index mapping fails

                         except ValueError:
                             df_processed_ohe_subset[col] = df_processed_ohe_subset[col].fillna('Missing') # Default to placeholder if column not found


            # Apply One-Hot Encoding using the aligned and prepared DataFrame subset
            encoded_data = onehot_encoder.transform(df_processed_ohe_subset)

            # Convert to dense array if sparse
            if isinstance(encoded_data, scipy.sparse.csr.csr_matrix):
                 encoded_data = encoded_data.toarray()

            # Create a DataFrame with encoded columns, ENSURING ALL EXPECTED OHE COLUMNS ARE PRESENT
            # Even if a category was not in the input, the column must exist with value 0.
            # Use the full list of expected OHE feature names from the encoder.
            temp_encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(ohe_cols_to_transform), index=df_processed.index)
            encoded_df = pd.DataFrame(0, index=df_processed.index, columns=full_expected_ohe_feature_names)

            # Copy data from the temporary DataFrame to the full DataFrame
            for col in temp_encoded_df.columns:
                 if col in encoded_df.columns:
                      encoded_df[col] = temp_encoded_df[col]


            # Drop original categorical columns (using the aligned list) from the main df_processed
            df_processed = df_processed.drop(columns=ohe_cols_to_transform)

            # Concatenate with encoded columns
            df_processed = pd.concat([df_processed, encoded_df], axis=1)

        except Exception as e:
             st.error(f"Could not apply One-Hot Encoding: {e}")
             # If OHE fails, drop original categorical columns to avoid errors later
             for col in ohe_cols_to_transform:
                 if col in df_processed.columns and not pd.api.types.is_numeric_dtype(df_processed[col]):
                      df_processed = df_processed.drop(columns=col)
             return None # Indicate failure if OHE fails


    # 4. Apply MinMax Scaling (ensure columns match the scaler's expected features)
    # Get the list of columns the scaler was fitted on
    scale_cols_expected_by_scaler = []
    if models_loaded and hasattr(loaded_scaler, 'feature_names_in_') and loaded_scaler.feature_names_in_ is not None:
        scale_cols_expected_by_scaler = list(loaded_scaler.feature_names_in_)
        # st.write(f"Scaler columns expected by scaler (from feature_names_in_): {scale_cols_expected_by_scaler}")
    else:
        st.error("Could not determine expected scaler columns from loaded_scaler.feature_names_in_. Cannot proceed with scaling.")
        return None

    # Create a temporary DataFrame with only the columns the scaler expects, from df_processed
    # Use reindex to ensure correct order and add missing columns with 0 (or another fill value)
    df_for_scaling = df_processed.reindex(columns=scale_cols_expected_by_scaler, fill_value=0)


    # Ensure all columns in the scaling subset are numeric before scaling
    for col in df_for_scaling.columns:
         if not pd.api.types.is_numeric_dtype(df_for_scaling[col]):
             try:
                 df_for_scaling[col] = pd.to_numeric(df_for_scaling[col], errors='coerce')
                 df_for_scaling[col] = df_for_scaling[col].fillna(0) # Fill NaNs after coercion
             except ValueError:
                  st.error(f"Column '{col}' is not numeric and could not be converted for scaling.")
                  return None # Indicate failure if a column cannot be made numeric


    # st.write("Columns being passed to Scaler:", df_for_scaling.columns.tolist())
    # st.write("Shape being passed to Scaler:", df_for_scaling.shape)
    # st.write("Expected Scaler shape:", loaded_scaler.n_features_in_)


    try:
        # Apply the scaler transformation
        scaled_data = loaded_scaler.transform(df_for_scaling)

        # Create a DataFrame from the scaled data, using the expected column names
        df_scaled = pd.DataFrame(scaled_data, columns=scale_cols_expected_by_scaler, index=df_processed.index)

        # Drop the original (unscaled) numerical columns from df_processed
        df_processed = df_processed.drop(columns=scale_cols_expected_by_scaler)

        # Concatenate with the scaled data
        df_processed = pd.concat([df_processed, df_scaled], axis=1)

    except Exception as e:
         st.error(f"Could not apply MinMax Scaler: {e}")
         return None # Indicate failure


    # 5. Align columns with expected features for PCA
    if models_loaded and EXPECTED_FEATURES_AFTER_PREPROCESSING is not None and len(EXPECTED_FEATURES_AFTER_PREPROCESSING) > 0:
        expected_features = EXPECTED_FEATURES_AFTER_PREPROCESSING

        # Create a new DataFrame with the expected columns and order, filling missing with 0
        df_aligned = pd.DataFrame(0, index=df_processed.index, columns=expected_features)

        # Copy data from df_processed to the new aligned DataFrame
        # Ensure all columns from df_processed that are in expected_features are copied
        cols_to_copy = [col for col in df_processed.columns if col in expected_features]
        df_aligned[cols_to_copy] = df_processed[cols_to_copy]


        # Ensure all columns are numeric before PCA and fill any remaining NaNs
        for col in df_aligned.columns:
             if not pd.api.types.is_numeric_dtype(df_aligned[col]):
                 try:
                     # Final attempt to convert to numeric and fill NaNs
                     df_aligned[col] = pd.to_numeric(df_aligned[col], errors='coerce')
                     df_aligned[col] = df_aligned[col].fillna(0)
                 except ValueError:
                     # st.warning(f"Column '{col}' is not numeric and could not be converted in final check. It will be dropped for PCA.")
                     df_aligned[col] = df_aligned[col].fillna(0) # Fill with 0 if conversion fails


        # Debugging checks before PCA
        # st.write("Columns in df_aligned before PCA:", df_aligned.columns.tolist())
        # st.write("Shape of df_aligned before PCA:", df_aligned.shape)
        # if hasattr(pca_model, 'feature_names_in_'):
        #      st.write("Expected features for PCA (from pca_model.feature_names_in_):", list(pca_model.feature_names_in_))
        #      st.write("Expected features for PCA (shape):", pca_model.feature_names_in_.shape)


        # Final check: ensure number of columns matches expected features for PCA
        if hasattr(pca_model, 'n_features_in_'):
             if df_aligned.shape[1] != pca_model.n_features_in_:
                 st.error(f"Column mismatch before PCA: DataFrame has {df_aligned.shape[1]} columns, but PCA expects {pca_model.n_features_in_}. Cannot apply PCA.")
                 # Optionally print column differences for debugging
                 # missing_from_expected_pca = set(EXPECTED_FEATURES_AFTER_PREPROCESSING) - set(df_aligned.columns)
                 # extra_in_aligned = set(df_aligned.columns) - set(EXPECTED_FEATURES_AFTER_PROCESSING)
                 # st.write(f"Features expected by PCA but missing from aligned DF: {missing_from_expected_pca}")
                 # st.write(f"Extra features in aligned DF not expected by PCA: {extra_in_aligned}")
                 return None

        # Check if there are any non-finite values and report
        if any(count > 0 for count in df_aligned.apply(lambda x: np.sum(~np.isfinite(x))).values): # Use .values to check the array
            st.error("Non-finite values detected in df_aligned before PCA. Cannot proceed with PCA.")
            return None

        # 6. Apply PCA
        # Convert to numpy array for PCA transformation to avoid potential pandas/sklearn mismatches
        X_aligned = df_aligned.values

        # Ensure the PCA model has feature names set if it expects them (might help with alignment)
        if hasattr(pca_model, 'feature_names_in_'):
             # Ensure expected_features is a numpy array of strings if needed by PCA model
             if isinstance(expected_features, np.ndarray):
                 pca_model.feature_names_in_ = expected_features
             else:
                 pca_model.feature_names_in_ = np.array(expected_features, dtype=object)


        try:
            df_pca = pca_model.transform(X_aligned)
            # Convert PCA result to DataFrame for consistency
            pca_column_names = [f'principal_component_{i+1}' for i in range(df_pca.shape[1])]
            df_pca = pd.DataFrame(df_pca, columns=pca_column_names, index=df_aligned.index)

        except Exception as e:
            st.error(f"Could not apply PCA: {e}")
            return None # Indicate failure

    else:
        st.warning("Expected features for PCA are not defined. PCA will not be applied.")
        # If PCA cannot be applied, return the processed DataFrame (without PCA)
        df_pca = df_processed.copy()


    return df_pca

# --- User Interface with Streamlit ---
st.title("Iowa House Price Prediction")

# Removed the initial description

# Use the function that generates widgets to get the input data
input_df = get_input_data_from_widgets()

# --- Make Predictions ---
if input_df is not None and models_loaded:
    # st.write("Preprocessing data and making predictions...") # Removed debug print

    try:
        processed_df = preprocess_input(input_df)

        # Perform predictions if the model is loaded and preprocessing was successful
        if 'loaded_model' in locals() and processed_df is not None:
            # Ensure that the number of features after preprocessing/PCA
            # matches what the loaded model expects
            if hasattr(loaded_model, 'n_features_in_'):
                if processed_df.shape[1] != loaded_model.n_features_in_:
                    st.error(f"The number of features after final preprocessing ({processed_df.shape[1]}) does not match the number expected by the model ({loaded_model.n_features_in_}). Prediction could not be made.")
                    # st.write("Features expected by the model:", loaded_model.feature_names_in_) # Removed debug print
                    # st.write("Features after final preprocessing:", processed_df.columns.tolist()) # Removed debug print

                else:
                    predictions = loaded_model.predict(processed_df)
                    st.subheader("Predicted House Value:")
                    st.success(f"The predicted value of the house is: ${predictions[0]:,.2f}")

            else:
                 st.warning("The loaded model does not have the 'n_features_in_' attribute. Could not verify feature consistency. Attempting to predict anyway...")
                 try:
                     predictions = loaded_model.predict(processed_df)
                     st.subheader("Predicted House Value:")
                     st.success(f"El valor predicho de la casa es: ${predictions[0]:,.2f}")

                 except Exception as e:
                     st.error(f"Error making predictions: {e}")

        else:
            # st.warning("Could not load the model or preprocess the data to make predictions.") # Removed debug print
            pass # Keep silent if preprocessing/model loading failed, error message is already shown

    except Exception as e:
        st.error(f"An error occurred during preprocessing or prediction: {e}")

elif not models_loaded:
    # st.warning("Models and preprocessors were not loaded correctly. Cannot make predictions.") # Removed debug print
    pass # Keep silent if model loading failed, error message is already shown
