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

    # Widgets for numerical variables (examples)
    widget_key = 'MSSubClass'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('MSSubClass', options=[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190], index=0, key=widget_key)

    widget_key = 'LotFrontage'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('LotFrontage', min_value=0.0, value=70.0, step=1.0, key=widget_key)

    widget_key = 'LotArea'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('LotArea', min_value=0, value=10500, step=100, key=widget_key)

    widget_key = 'OverallQual'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.slider('OverallQual', 1, 10, 5, key=widget_key)

    widget_key = 'OverallCond'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.slider('OverallCond', 1, 9, 5, key=widget_key)

    widget_key = 'YearBuilt'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('YearBuilt', min_value=1800, value=1980, step=1, key=widget_key)

    widget_key = 'YearRemodAdd'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('YearRemodAdd', min_value=1800, value=1980, step=1, key=widget_key)

    widget_key = 'MasVnrArea'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('MasVnrArea', min_value=0.0, value=0.0, step=10.0, key=widget_key)

    widget_key = 'BsmtFinSF1'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BsmtFinSF1', min_value=0, value=500, step=10, key=widget_key)

    widget_key = 'BsmtFinSF2'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BsmtFinSF2', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'BsmtUnfSF'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BsmtUnfSF', min_value=0, value=500, step=10, key=widget_key)

    widget_key = 'TotalBsmtSF'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('TotalBsmtSF', min_value=0, value=1000, step=10, key=widget_key)

    widget_key = '1stFlrSF'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('1stFlrSF', min_value=0, value=1000, step=10, key=widget_key)

    widget_key = '2ndFlrSF'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('2ndFlrSF', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'LowQualFinSF'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('LowQualFinSF', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'GrLivArea'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('GrLivArea', min_value=0, value=1000, step=10, key=widget_key)

    widget_key = 'BsmtFullBath'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BsmtFullBath', min_value=0, value=0, step=1, key=widget_key)

    widget_key = 'BsmtHalfBath'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BsmtHalfBath', min_value=0, value=0, step=1, key=widget_key)

    widget_key = 'FullBath'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('FullBath', min_value=0, value=1, step=1, key=widget_key)

    widget_key = 'HalfBath'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('HalfBath', min_value=0, value=0, step=1, key=widget_key)

    widget_key = 'BedroomAbvGr'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BedroomAbvGr', min_value=0, value=3, step=1, key=widget_key)

    widget_key = 'KitchenAbvGr'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('KitchenAbvGr', min_value=0, value=1, step=1, key=widget_key)

    widget_key = 'TotRmsAbvGrd'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('TotRmsAbvGrd', min_value=0, value=5, step=1, key=widget_key)

    widget_key = 'Fireplaces'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('Fireplaces', min_value=0, value=0, step=1, key=widget_key)

    widget_key = 'GarageType'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('GarageType', options=['Attchd', 'Detchd', 'BuiltIn', 'CarPort', np.nan, 'Basment', '2Types'], index=0, key=widget_key)

    widget_key = 'GarageYrBlt'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('GarageYrBlt', min_value=1800.0, value=1980.0, step=1.0, key=widget_key)

    widget_key = 'GarageCars'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('GarageCars', min_value=0, value=1, step=1, key=widget_key)

    widget_key = 'GarageArea'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('GarageArea', min_value=0, value=300, step=10, key=widget_key)

    widget_key = 'WoodDeckSF'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('WoodDeckSF', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'OpenPorchSF'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('OpenPorchSF', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'EnclosedPorch'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('EnclosedPorch', min_value=0, value=0, step=10, key=widget_key)

    widget_key = '3SsnPorch'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('3SsnPorch', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'ScreenPorch'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('ScreenPorch', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'PoolArea'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('PoolArea', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'PoolQC'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('PoolQC', options=[np.nan, 'Ex', 'Gd', 'TA', 'Fa'], index=1, key=widget_key)

    widget_key = 'Fence'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Fence', options=[np.nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'], index=1, key=widget_key)

    widget_key = 'MiscFeature'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('MiscFeature', options=[np.nan, 'Shed', 'Gar2', 'Othr', 'TenC'], index=1, key=widget_key)

    widget_key = 'MiscVal'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('MiscVal', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'MoSold'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.slider('MoSold', 1, 12, 6, key=widget_key)

    widget_key = 'YrSold'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.slider('YrSold', 2006, 2010, 2008, key=widget_key)

    # Id is not a Streamlit widget and does not require a key.
    input_data['Id'] = 0 # A placeholder ID

    # Widgets for categorical variables (examples with common options)
    widget_key = 'MSZoning'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('MSZoning', options=['RL', 'RM', 'C (all)', 'FV', 'RH'], index=0, key=widget_key)

    widget_key = 'Street'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Street', options=['Pave', 'Grvl'], index=0, key=widget_key)

    widget_key = 'Alley'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Alley', options=[np.nan, 'Grvl', 'Pave'], index=1, key=widget_key)

    widget_key = 'LotShape'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('LotShape', options=['Reg', 'IR1', 'IR2', 'IR3'], index=0, key=widget_key)

    widget_key = 'LandContour'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('LandContour', options=['Lvl', 'Bnk', 'HLS', 'Low'], index=0, key=widget_key)

    widget_key = 'Utilities'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Utilities', options=['AllPub', 'NoSeWa'], index=0, key=widget_key)

    widget_key = 'LotConfig'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('LotConfig', options=['Inside', 'Frontage', 'Corner', 'CulDSac', 'FR2', 'FR3'], index=0, key=widget_key)

    widget_key = 'LandSlope'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('LandSlope', options=['Gtl', 'Mod', 'Sev'], index=0, key=widget_key)

    widget_key = 'Neighborhood'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Neighborhood', options=['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber', 'IDOTC', 'ClearCr', 'StoneBr', 'SWISU', 'Blmngtn', 'MeadowV', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'], index=0, key=widget_key)

    widget_key = 'Condition1'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Condition1', options=['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe'], index=0, key=widget_key)

    widget_key = 'Condition2'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Condition2', options=['Norm', 'Artery', 'Feedr', 'RRNn', 'RRAn', 'RRAe', 'PosN', 'PosA'], index=0, key=widget_key)

    widget_key = 'BldgType'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BldgType', options=['1Fam', '2FmCon', 'Duplex', 'TwnhsE', 'Twnhs'], index=0, key=widget_key)

    widget_key = 'HouseStyle'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('HouseStyle', options=['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'], index=0, key=widget_key)

    widget_key = 'RoofStyle'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('RoofStyle', options=['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed'], index=0, key=widget_key)

    widget_key = 'RoofMatl'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('RoofMatl', options=['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Roll', 'ClyTile'], index=0, key=widget_key)

    widget_key = 'Exterior1st'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Exterior1st', options=['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'CBlock', 'BrkComm', 'AsphShn', 'VinylSd', 'ImStucc', 'Stone', 'Other'], index=0, key=widget_key)

    widget_key = 'Exterior2nd'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Exterior2nd', options=['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'BrkFace', 'Wd Sdng', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'CBlock', 'BrkComm', 'AsphShn', 'VinylSd', 'ImStucc', 'Stone', 'Other'], index=0, key=widget_key)

    widget_key = 'MasVnrType'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('MasVnrType', options=['None', 'BrkFace', 'Stone', 'BrkCmn', np.nan], index=1, key=widget_key)

    widget_key = 'Foundation'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Foundation', options=['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'], index=0, key=widget_key)

    widget_key = 'BsmtQual'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtQual', options=['Gd', 'TA', 'Ex', 'Fa', np.nan], index=1, key=widget_key) # Use index=1 for 'TA' as default

    widget_key = 'BsmtCond'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtCond', options=['TA', 'Gd', 'Fa', 'Po', np.nan], index=0, key=widget_key) # Use index=0 for 'TA' as default

    widget_key = 'BsmtExposure'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtExposure', options=['No', 'Gd', 'Mn', 'Av', np.nan], index=0, key=widget_key)

    widget_key = 'BsmtFinType1'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtFinType1', options=['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ', np.nan], index=1, key=widget_key) # Use index=1 for 'ALQ' as default

    widget_key = 'BsmtFinType2'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtFinType2', options=['Unf', 'Rec', 'LwQ', 'GLQ', 'ALQ', 'BLQ', np.nan], index=0, key=widget_key) # Use index=0 for 'Unf' as default

    widget_key = 'Heating'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Heating', options=['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'], index=0, key=widget_key)

    widget_key = 'HeatingQC'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('HeatingQC', options=['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=0, key=widget_key) # Use index=0 for 'Ex' as default

    widget_key = 'CentralAir'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('CentralAir', options=['Y', 'N'], index=0, key=widget_key)

    widget_key = 'Electrical'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Electrical', options=['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', np.nan], index=1, key=widget_key)

    widget_key = 'KitchenQual'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('KitchenQual', options=['Gd', 'TA', 'Ex', 'Fa'], index=1, key=widget_key) # Use index=1 for 'TA' as default

    widget_key = 'Functional'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Functional', options=['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev'], index=0, key=widget_key) # Use index=0 for 'Typ' as default

    widget_key = 'FireplaceQu'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('FireplaceQu', options=[np.nan, 'Gd', 'TA', 'Fa', 'Ex', 'Po'], index=1, key=widget_key)

    # Convert the dictionary to DataFrame
    return pd.DataFrame([input_data])


# --- Function to Preprocess Input Data ---
def preprocess_input(input_df):
    """Applies the same preprocessing steps used in training."""
    df_processed = input_df.copy()

    # Ensure all expected original columns exist in the input DataFrame
    # This is crucial for consistent preprocessing
    expected_original_cols = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'Exterior1st', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition']

    # Add missing columns with NaN
    for col in expected_original_cols:
        if col not in df_processed.columns:
            df_processed[col] = np.nan

    # Select only the expected original columns in the correct order
    df_processed = df_processed[expected_original_cols].copy()


    # 1. Convert to category if necessary (ensure columns exist)
    current_categorical_cols = df_processed.select_dtypes(include='object').columns
    for col in current_categorical_cols:
         if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')


    # 2. Apply LabelEncoder (handle unseen labels)
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
        if col in df_processed.columns and pd.api.types.is_categorical_dtype(df_processed[col]):
             encoder = label_encoders[col]
             try:
                 # Handle unseen labels: replace with a label the encoder knows or NaN
                 df_processed[col] = df_processed[col].astype(pd.CategoricalDtype(categories=encoder.classes_))
                 df_processed[col] = df_processed[col].cat.codes
                 df_processed[col] = df_processed[col].replace(-1, np.nan) # Convert -1 from cat.codes (unseen) to NaN
                 df_processed[col] = df_processed[col].fillna(0) # Fill NaN with 0 (or another strategy)
             except Exception as e:
                 st.warning(f"Could not apply LabelEncoder to column '{col}': {e}")
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(0)


    # Convert LabelEncoder columns (now numeric) to numeric type if they are not already
    for col in label_encode_cols:
        if col in df_processed.columns:
             if not pd.api.types.is_numeric_dtype(df_processed[col]):
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(0)


    # 3. Apply One-Hot Encoding (ensure columns exist and handle missing/extra columns)
    categorical_cols_ohe = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'Exterior1st', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition']
    ohe_cols_in_df = [col for col in categorical_cols_ohe if col in df_processed.columns]

    if ohe_cols_in_df:
        try:
            # Ensure handle_unknown='ignore' is set for the onehot_encoder
            if hasattr(onehot_encoder, 'handle_unknown'):
                 onehot_encoder.handle_unknown = 'ignore'
            else:
                st.warning("OneHotEncoder does not support handle_unknown='ignore'. Unseen categories may cause errors.")


            # Handle problematic categories and fill NaN values in categorical columns
            for col in ohe_cols_in_df:
                if col in df_processed.columns:
                    try:
                        col_index_in_encoder = categorical_cols_ohe.index(col)
                        encoder_categories = onehot_encoder.categories_[col_index_in_encoder]

                        # Find the first non-NaN, non-problematic category from the encoder's categories
                        # Exclude 'C (all)' and any other categories that caused issues
                        problematic_categories = ['Missing', 'C (all)', 'FV'] # Add any other categories that cause errors
                        first_valid_category = next((cat for cat in encoder_categories if pd.notna(cat) and cat not in problematic_categories), None)

                        fill_value = str(first_valid_category) if first_valid_category is not None else 'UnknownCategory' # Use a distinct placeholder

                        # Replace any value not in encoder's categories (excluding NaN) with the fill_value
                        valid_encoder_categories_list = [str(cat) for cat in encoder_categories if pd.notna(cat)]
                        df_processed[col] = df_processed[col].apply(lambda x: fill_value if pd.notna(x) and str(x) not in valid_encoder_categories_list else x)

                        df_processed[col] = df_processed[col].fillna(fill_value).astype(str) # Fill remaining NaN and ensure string type

                    except ValueError:
                         st.warning(f"Column '{col}' not found in onehot_encoder categories. Skipping specific handling.")
                         df_processed[col] = df_processed[col].fillna('UnknownCategory').astype(str) # Fallback fill if column not in encoder


            # Apply One-Hot Encoding
            encoded_data = onehot_encoder.transform(df_processed[ohe_cols_in_df])


            # Convert to dense array if sparse
            if isinstance(encoded_data, scipy.sparse.csr.csr_matrix):
                 encoded_data = encoded_data.toarray()

            # Create a DataFrame with encoded columns
            encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(ohe_cols_in_df), index=df_processed.index)

            # Drop original categorical columns
            df_processed = df_processed.drop(columns=ohe_cols_in_df)

            # Concatenate with encoded columns
            df_processed = pd.concat([df_processed, encoded_df], axis=1)
        except Exception as e:
             st.error(f"Could not apply One-Hot Encoding: {e}")
             # If OHE fails, drop categorical columns to avoid errors later
             for col in ohe_cols_in_df:
                 if col in df_processed.columns and not pd.api.types.is_numeric_dtype(df_processed[col]):
                      df_processed = df_processed.drop(columns=col)
             return None # Indicate failure if OHE fails


    # 4. Scale numerical columns (ensure columns exist and handle NaNs)
    numerical_cols_to_scale = ['OverallQual', 'OverallCond','LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF']
    scale_cols_in_df = [col for col in numerical_cols_to_scale if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]

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
                df_aligned[col] = df_processed[col]

        # Ensure all columns are numeric before PCA and fill any remaining NaNs
        for col in df_aligned.columns:
             if not pd.api.types.is_numeric_dtype(df_aligned[col]):
                 try:
                     df_aligned[col] = pd.to_numeric(df_aligned[col])
                     df_aligned[col] = df_aligned[col].fillna(0)
                 except ValueError:
                     st.warning(f"Column '{col}' is not numeric and could not be converted. It will be dropped for PCA.")
                     df_aligned = df_aligned.drop(columns=col)
             else:
                 df_aligned[col] = df_aligned[col].fillna(0)

        # Debugging checks before PCA
        # st.write("Columns in df_aligned before PCA:", df_aligned.columns.tolist())
        # st.write("Data types in df_aligned before PCA:", df_aligned.dtypes)
        # st.write("Checking for non-finite values in df_aligned before PCA:")
        # non_finite_counts = df_aligned.apply(lambda x: np.sum(~np.isfinite(x))).to_dict()
        # st.write(non_finite_counts)

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
                    st.error(f"The number of features after preprocessing ({processed_df.shape[1]}) does not match the number expected by the model ({loaded_model.n_features_in_}). Prediction could not be made.")
                    st.write("Features expected by the model:", loaded_model.feature_names_in_)
                    st.write("Features after preprocessing:", processed_df.columns.tolist())

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
