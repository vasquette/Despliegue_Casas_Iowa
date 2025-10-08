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
            # Determine default index (try to find 'None' or first option if nan is not an option)
            try:
                 default_index = options.index('None') if 'None' in options else (options.index(np.nan) if np.nan in options else 0)
            except ValueError:
                 default_index = 0 # Default to first option if None or nan not found

            input_data[widget_key] = st.sidebar.selectbox(col, options=options, index=default_index, key=widget_key)
        elif col in numerical_defaults:
            default_value = numerical_defaults[col]
            if isinstance(default_value, int):
                 input_data[widget_key] = st.sidebar.number_input(col, min_value=0, value=default_value, step=1, key=widget_key)
            else: # Assume float
                 input_data[widget_key] = st.sidebar.number_input(col, min_value=0.0, value=default_value, step=10.0, key=widget_key)
        else:
            # Fallback for any other columns (e.g., boolean or less common types)
            st.sidebar.warning(f"No specific widget defined for column: {col}. Using text input as fallback.")
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
        'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath',
        'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF',
        # Categorical columns that might be OHE or Label Encoded
        'MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1',
        'BldgType', 'HouseStyle', 'Exterior1st', 'ExterQual', 'ExterCond', 'Foundation',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
        'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish',
        'PavedDrive', 'SaleType', 'SaleCondition', 'Street', 'Alley', 'Utilities',
        'LandSlope', 'Condition2', 'RoofStyle', 'RoofMatl', 'Exterior2nd', 'MasVnrType',
        'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'GarageYrBlt', 'MiscVal',
        'MoSold', 'YrSold' # Numerical columns used later
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
                     df_processed[col] = df_processed[col].astype(str).replace('nan', np.nan) # Convert 'nan' string back to NaN before astype

                 # Handle unseen labels: convert to CategoricalDtype with known categories
                 # Fill NaNs before converting to categorical to avoid them being treated as a new category
                 df_processed[col] = df_processed[col].fillna('Missing').astype(pd.CategoricalDtype(categories=list(encoder.classes_) + ['Missing']))
                 df_processed[col] = df_processed[col].cat.codes
                 df_processed[col] = df_processed[col].replace(-1, np.nan) # Convert -1 from cat.codes (unseen) to NaN
                 df_processed[col] = df_processed[col].fillna(0) # Fill NaNs resulting from unseen/missing

             except Exception as e:
                 st.warning(f"Could not apply LabelEncoder to column '{col}': {e}")
                 # Fallback: convert to numeric with errors='coerce' and fill NaNs
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(0)


    # Convert LabelEncoder columns (now numeric) to numeric type if they are not already
    for col in label_encode_cols:
        if col in df_processed.columns:
             if not pd.api.types.is_numeric_dtype(df_processed[col]):
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(0)


    # 3. Apply One-Hot Encoding (Dynamically determine columns from encoder)
    ohe_cols_to_transform = []
    if models_loaded and hasattr(onehot_encoder, 'feature_names_in_') and onehot_encoder.feature_names_in_ is not None:
        # Extract original column names from encoder's feature_names_in_
        # These names have the format 'original_col__category'
        ohe_cols_expected_by_encoder = [col.split('__')[0] for col in onehot_encoder.get_feature_names_out()] # Use get_feature_names_out to get resulting names
        # Get unique original column names while preserving order
        unique_ohe_original_cols_expected = []
        for col in ohe_cols_expected_by_encoder:
            if col not in unique_ohe_original_cols_expected:
                unique_ohe_original_cols_expected.append(col)

        ohe_cols_to_transform = [col for col in unique_ohe_original_cols_expected if col in df_processed.columns]
        st.write(f"OHE columns expected by encoder (inferred from get_feature_names_out): {unique_ohe_original_cols_expected}")
        st.write(f"OHE columns found in input DF for transformation: {ohe_cols_to_transform}")

        # Verify that all expected OHE columns are present in the input DataFrame
        if len(ohe_cols_to_transform) != len(unique_ohe_original_cols_expected):
            missing_cols = set(unique_ohe_original_cols_expected) - set(ohe_cols_to_transform)
            st.error(f"Preprocessing error: Input DataFrame is missing expected categorical columns for One-Hot Encoding: {missing_cols}")
            st.write(f"Expected OHE original columns: {unique_ohe_original_cols_expected}")
            st.write(f"Available columns in input DF: {df_processed.columns.tolist()}")
            return None # Indicate failure if columns are missing


    elif models_loaded and hasattr(onehot_encoder, 'categories_'):
         st.warning("OneHotEncoder does not have 'feature_names_in_'. Attempting to infer columns from categories_ structure.")
         # If feature_names_in_ is not available, try to use categories_ length
         # This is less reliable as it doesn't give original column names
         # Based on the user's previous output, there are 19 category sets.
         # Let's assume the 19 columns are the initial ones from a typical list.
         # This is a fallback and might require manual adjustment if it fails.
         try:
             # This is a fragile assumption based on previous error output and common dataset knowledge
             ohe_cols_to_transform = [
                'MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood',
                'Condition1', 'BldgType', 'HouseStyle', 'Exterior1st', 'Exterior2nd',
                'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'CentralAir', 'Electrical', 'GarageType', 'SaleType' # Assuming these 19
             ]
             ohe_cols_to_transform = [col for col in ohe_cols_to_transform if col in df_processed.columns]
             if len(ohe_cols_to_transform) != len(onehot_encoder.categories_):
                 st.error(f"Preprocessing error: Number of columns ({len(ohe_cols_to_transform)}) inferred for OHE does not match the number of categories sets in the encoder ({len(onehot_encoder.categories_)}). Check the list of OHE_EXPECTED_COLS_FROM_ENCODER.")
                 return None # Indicate failure

             st.write(f"OHE columns inferred from categories_ (length {len(ohe_cols_to_transform)}): {ohe_cols_to_transform}")

         except Exception as e:
             st.error(f"Could not infer expected OHE columns from encoder.categories_: {e}. Cannot proceed with OHE.")
             return None # Indicate failure

    else:
        st.error("Could not determine expected OHE columns from the encoder. Cannot proceed with OHE.")
        return None # Indicate failure


    if ohe_cols_to_transform:
        # Ensure the columns are in the correct order for transformation
        df_processed_ohe_subset = df_processed[ohe_cols_to_transform].copy()

        st.write(f"Columns being passed to OneHotEncoder: {df_processed_ohe_subset.columns.tolist()}")
        st.write(f"Data types of columns being passed to OneHotEncoder: {df_processed_ohe_subset.dtypes}")
        st.write(f"Sample data for OHE columns:\n{df_processed_ohe_subset.head()}")
        if models_loaded and hasattr(onehot_encoder, 'categories_'):
            st.write(f"Encoder's categories structure (length {len(onehot_encoder.categories_)}):")
            # Match category index to column name based on ohe_cols_to_transform
            for i, cats in enumerate(onehot_encoder.categories_):
                col_name = ohe_cols_to_transform[i] if i < len(ohe_cols_to_transform) else f"Unknown_col_{i}"
                st.write(f"  Column '{col_name}' (Index {i}): {list(cats)}")


        try:
            # Ensure handle_unknown='ignore' is set for the onehot_encoder
            if hasattr(onehot_encoder, 'handle_unknown'):
                 onehot_encoder.handle_unknown = 'ignore'
                 st.write("OneHotEncoder handle_unknown is set to 'ignore'.")
            else:
                st.warning("OneHotEncoder does not support handle_unknown='ignore'. Manual handling of unseen categories is crucial.")


            # Convert columns to string type to avoid issues with mixed types before OHE
            for col in ohe_cols_to_transform:
                 if col in df_processed_ohe_subset.columns:
                     # Convert any non-string/non-NaN values to string, handle NaNs
                     df_processed_ohe_subset[col] = df_processed_ohe_subset[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)
                     # Fill remaining NaNs with a placeholder if 'nan' is not an expected category
                     if df_processed_ohe_subset[col].isnull().any():
                         # Try to get categories for this column from the encoder to see if 'nan' is expected
                         col_index_in_encoder = ohe_cols_to_transform.index(col) # Use index in the current subset
                         encoder_categories_for_col = [str(cat) for cat in onehot_encoder.categories_[col_index_in_encoder] if pd.notna(cat)]
                         if 'nan' not in encoder_categories_for_col:
                             df_processed_ohe_subset[col] = df_processed_ohe_subset[col].fillna('Missing') # Use 'Missing' as a consistent placeholder
                         else:
                              df_processed_ohe_subset[col] = df_processed_ohe_subset[col].fillna('nan') # Fill with 'nan' string if encoder expects it


            # Apply One-Hot Encoding using the aligned and prepared DataFrame subset
            encoded_data = onehot_encoder.transform(df_processed_ohe_subset)

            # Convert to dense array if sparse
            if isinstance(encoded_data, scipy.sparse.csr.csr_matrix):
                 encoded_data = encoded_data.toarray()

            # Create a DataFrame with encoded columns
            # Use get_feature_names_out without passing input features to get all resulting column names
            encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(), index=df_processed.index)

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


    # 4. Scale numerical columns (ensure columns exist and handle NaNs)
    # Update numerical columns list to potentially include columns previously OHE but now handled differently
    numerical_cols_to_scale = ['OverallQual', 'OverallCond','LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'LotFrontage'] # Added more numerical columns
    # Add columns that were label encoded to the list of numerical columns to fillna and scale
    label_encode_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'Functional', 'GarageFinish'] # Redefine for clarity
    all_numeric_cols = list(set(numerical_cols_to_scale + label_encode_cols)) # Combine lists and remove duplicates

    scale_cols_in_df = [col for col in all_numeric_cols if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]

    if scale_cols_in_df:
        df_processed[scale_cols_in_df] = df_processed[scale_cols_in_df].fillna(0) # Fill NaNs before scaling
        try:
            df_processed[scale_cols_in_df] = loaded_scaler.transform(df_processed[scale_cols_in_df])
        except Exception as e:
             st.warning(f"Could not apply MinMax Scaler: {e}")


    # 5. Align columns with expected features for PCA
    if EXPECTED_FEATURES_AFTER_PREPROCESSING is not None and len(EXPECTED_FEATURES_AFTER_PREPROCESSING) > 0:
        expected_features = EXPECTED_FEATURES_AFTER_PREPROCESSING

        # Create a new DataFrame with the expected columns and order, filling missing with 0
        df_aligned = pd.DataFrame(0, index=df_processed.index, columns=expected_features)

        # Copy data from df_processed to the new aligned DataFrame
        for col in df_aligned.columns:
            if col in df_processed.columns:
                 # Ensure the data type is compatible before assignment
                 if pd.api.types.is_numeric_dtype(df_processed[col]):
                      df_aligned[col] = df_processed[col]
                 else:
                     # Attempt conversion to numeric if it's not, with error handling
                     try:
                         df_aligned[col] = pd.to_numeric(df_processed[col])
                     except ValueError:
                         st.warning(f"Column '{col}' could not be converted to numeric during final alignment. Filling with 0.")
                         df_aligned[col] = 0


        # Ensure all columns are numeric before PCA and fill any remaining NaNs
        for col in df_aligned.columns:
             if not pd.api.types.is_numeric_dtype(df_aligned[col]):
                 try:
                     # Final attempt to convert to numeric and fill NaNs
                     df_aligned[col] = pd.to_numeric(df_aligned[col], errors='coerce')
                     df_aligned[col] = df_aligned[col].fillna(0)
                 except ValueError:
                     st.warning(f"Column '{col}' is not numeric and could not be converted in final check. It will be dropped for PCA.")
                     df_aligned = df_aligned.drop(columns=col)
             else:
                 df_aligned[col] = df_aligned[col].fillna(0)

        # Debugging checks before PCA
        st.write("Columns in df_aligned before PCA:", df_aligned.columns.tolist())
        st.write("Shape of df_aligned before PCA:", df_aligned.shape)
        if hasattr(pca_model, 'feature_names_in_'):
             st.write("Expected features for PCA (from pca_model.feature_names_in_):", list(pca_model.feature_names_in_))
             st.write("Expected features for PCA (shape):", pca_model.feature_names_in_.shape)


        # Final check: ensure number of columns matches expected features for PCA
        if hasattr(pca_model, 'n_features_in_'):
             if df_aligned.shape[1] != pca_model.n_features_in_:
                 st.error(f"Column mismatch before PCA: DataFrame has {df_aligned.shape[1]} columns, but PCA expects {pca_model.n_features_in_}. Cannot apply PCA.")
                 # Optionally print column differences for debugging
                 missing_from_expected_pca = set(EXPECTED_FEATURES_AFTER_PREPROCESSING) - set(df_aligned.columns)
                 extra_in_aligned = set(df_aligned.columns) - set(EXPECTED_FEATURES_AFTER_PREPROCESSING)
                 st.write(f"Features expected by PCA but missing from aligned DF: {missing_from_expected_pca}")
                 st.write(f"Extra features in aligned DF not expected by PCA: {extra_in_aligned}")
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

st.write("This application predicts house prices based on their characteristics. Enter the characteristic values below to get a prediction.")

# Use the function that generates widgets to get the input data
input_df = get_input_data_from_widgets()

# --- Make Predictions ---
if input_df is not None and models_loaded:
    st.write("Preprocessing data and making predictions...")

    try:
        processed_df = preprocess_input(input_df)

        # Perform predictions if the model is loaded and preprocessing was successful
        if 'loaded_model' in locals() and processed_df is not None:
            # Ensure that the number of features after preprocessing/PCA
            # matches what the loaded model expects
            if hasattr(loaded_model, 'n_features_in_'):
                if processed_df.shape[1] != loaded_model.n_features_in_:
                    st.error(f"The number of features after final preprocessing ({processed_df.shape[1]}) does not match the number expected by the model ({loaded_model.n_features_in_}). Prediction could not be made.")
                    st.write("Features expected by the model:", loaded_model.feature_names_in_)
                    st.write("Features after final preprocessing:", processed_df.columns.tolist())

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
            st.warning("Could not load the model or preprocess the data to make predictions.")

    except Exception as e:
        st.error(f"An error occurred during preprocessing or prediction: {e}")

elif not models_loaded:
    st.warning("Models and preprocessors were not loaded correctly. Cannot make predictions.")
