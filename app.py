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

    # Define descriptions for each feature
    feature_descriptions = {
        'MSSubClass': 'Identifies the type of dwelling involved in the sale.',
        'MSZoning': 'Identifies the general zoning classification of the sale.',
        'LotFrontage': 'Linear feet of street connected to property.',
        'LotArea': 'Lot size in square feet.',
        'Street': 'Type of road access to property.',
        'Alley': 'Type of alley access to property.',
        'LotShape': 'General shape of property.',
        'LandContour': 'Flatness of the property.',
        'Utilities': 'Type of utilities available.',
        'LotConfig': 'Lot configuration.',
        'LandSlope': 'Slope of property.',
        'Neighborhood': 'Physical locations within Ames city limits.',
        'Condition1': 'Proximity to main road or railroad.',
        'Condition2': 'Proximity to main road or railroad (if a second is present).',
        'BldgType': 'Type of dwelling.',
        'HouseStyle': 'Style of dwelling.',
        'OverallQual': 'Rates the overall material and finish of the house.',
        'OverallCond': 'Rates the overall condition of the house.',
        'YearBuilt': 'Original construction date.',
        'YearRemodAdd': 'Remodel date (same as construction date if no remodel or addition).',
        'RoofStyle': 'Type of roof.',
        'RoofMatl': 'Roofing material.',
        'Exterior1st': 'Exterior covering on house.',
        'Exterior2nd': 'Exterior covering on house (if more than one material).',
        'MasVnrType': 'Masonry veneer type.',
        'MasVnrArea': 'Masonry veneer area in square feet.',
        'ExterQual': 'Evaluates the quality of the material on the exterior.',
        'ExterCond': 'Evaluates the present condition of the material on the exterior.',
        'Foundation': 'Type of foundation.',
        'BsmtQual': 'Evaluates the height of the basement.',
        'BsmtCond': 'Evaluates the general condition of the basement.',
        'BsmtExposure': 'Refers to walkout or garden level walls.',
        'BsmtFinType1': 'Rating of basement finished area (Type 1).',
        'BsmtFinSF1': 'Type 1 finished square feet.',
        'BsmtFinType2': 'Rating of basement finished area (Type 2).',
        'BsmtFinSF2': 'Type 2 finished square feet.',
        'BsmtUnfSF': 'Unfinished square feet of basement area.',
        'TotalBsmtSF': 'Total square feet of basement area.',
        'Heating': 'Type of heating.',
        'HeatingQC': 'Heating quality and condition.',
        'CentralAir': 'Central air conditioning.',
        'Electrical': 'Electrical system.',
        '1stFlrSF': 'First Floor square feet.',
        '2ndFlrSF': 'Second floor square feet.',
        'LowQualFinSF': 'Low quality finished square feet (all floors).',
        'GrLivArea': 'Above grade (ground) living area square feet.',
        'BsmtFullBath': 'Basement full bathrooms.',
        'BsmtHalfBath': 'Basement half bathrooms.',
        'FullBath': 'Full bathrooms above grade.',
        'HalfBath': 'Half baths above grade.',
        'BedroomAbvGr': 'Bedrooms above grade (does not include basement bedrooms).',
        'KitchenAbvGr': 'Kitchens above grade.',
        'KitchenQual': 'Kitchen quality.',
        'TotRmsAbvGrd': 'Total rooms above grade (does not include basement bathrooms).',
        'Functional': 'Home functionality (assume typical unless otherwise noted).',
        'Fireplaces': 'Number of fireplaces.',
        'FireplaceQu': 'Fireplace quality.',
        'GarageType': 'Garage location.',
        'GarageYrBlt': 'Year garage was built.',
        'GarageFinish': 'Interior finish of the garage.',
        'GarageCars': 'Size of garage in car capacity.',
        'GarageArea': 'Size of garage in square feet.',
        'PavedDrive': 'Paved driveway.',
        'WoodDeckSF': 'Wood deck area in square feet.',
        'OpenPorchSF': 'Open porch area in square feet.',
        'EnclosedPorch': 'Enclosed porch area in square feet.',
        '3SsnPorch': 'Three season porch area in square feet.',
        'ScreenPorch': 'Screen porch area in square feet.',
        'PoolArea': 'Pool area in square feet.',
        'PoolQC': 'Pool quality.',
        'Fence': 'Fence quality.',
        'MiscFeature': 'Miscellaneous feature not covered in other categories.',
        'MiscVal': 'Value of miscellaneous feature.',
        'MoSold': 'Month Sold (MM).',
        'YrSold': 'Year Sold (YYYY).',
        'SaleType': 'Type of sale.',
        'SaleCondition': 'Condition of sale.'
    }


    # --- Organize Widgets into Sections ---

    # General Property Characteristics
    st.sidebar.subheader('General Property Characteristics')
    col = 'MSSubClass'
    input_data[col] = st.sidebar.selectbox(col, options=[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190], index=0, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'MSZoning'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('RL'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'LotFrontage'
    input_data[col] = st.sidebar.number_input(col, min_value=0.0, value=70.0, step=1.0, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'LotArea'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=10500, step=100, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Street'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Pave'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Alley'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index(np.nan) if np.nan in categorical_options[col] else 0, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'LotShape'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Reg'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'LandContour'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Lvl'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Utilities'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('AllPub'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'LotConfig'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Inside'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'LandSlope'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Gtl'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Neighborhood'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('NAmes'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Condition1'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Norm'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Condition2'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Norm'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'BldgType'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('1Fam'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'HouseStyle'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('1Story'), key=col, help=feature_descriptions.get(col, 'No description available.'))

    # Quality and Condition
    st.sidebar.subheader('Quality and Condition')
    col = 'OverallQual'
    input_data[col] = st.sidebar.slider(col, 1, 10, 5, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'OverallCond'
    input_data[col] = st.sidebar.slider(col, 1, 9, 5, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'ExterQual'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('TA'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'ExterCond'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('TA'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'HeatingQC'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Ex'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'KitchenQual'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('TA'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Functional'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Typ'), key=col, help=feature_descriptions.get(col, 'No description available.'))


    # Construction and Remodeling
    st.sidebar.subheader('Construction and Remodeling')
    col = 'YearBuilt'
    input_data[col] = st.sidebar.number_input(col, min_value=1800, value=1980, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'YearRemodAdd'
    input_data[col] = st.sidebar.number_input(col, min_value=1800, value=1980, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'RoofStyle'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Gable'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'RoofMatl'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('CompShg'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Exterior1st'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('VinylSd'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Exterior2nd'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('VinylSd'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'MasVnrType'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('None') if 'None' in categorical_options[col] else 0, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'MasVnrArea'
    input_data[col] = st.sidebar.number_input(col, min_value=0.0, value=0.0, step=10.0, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Foundation'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('PConc'), key=col, help=feature_descriptions.get(col, 'No description available.'))


    # Basement Features
    st.sidebar.subheader('Basement Features')
    col = 'BsmtQual'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('TA'), key=col, help=feature_descriptions.get(col, 'No description available.')) # Use index for 'TA'
    col = 'BsmtCond'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('TA'), key=col, help=feature_descriptions.get(col, 'No description available.')) # Use index for 'TA'
    col = 'BsmtExposure'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('No'), key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'BsmtFinType1'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Unf'), key=col, help=feature_descriptions.get(col, 'No description available.')) # Use index for 'Unf'
    col = 'BsmtFinSF1'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=500, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'BsmtFinType2'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Unf'), key=col, help=feature_descriptions.get(col, 'No description available.')) # Use index for 'Unf'
    col = 'BsmtFinSF2'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'BsmtUnfSF'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=500, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'TotalBsmtSF'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=1000, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'BsmtFullBath'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'BsmtHalfBath'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))


    # Above Ground Features
    st.sidebar.subheader('Above Ground Features')
    col = '1stFlrSF'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=1000, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = '2ndFlrSF'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'LowQualFinSF'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'GrLivArea'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=1000, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'FullBath'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=1, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'HalfBath'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'BedroomAbvGr'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=3, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'KitchenAbvGr'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=1, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'TotRmsAbvGrd'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=5, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Fireplaces'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'FireplaceQu'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index(np.nan) if np.nan in categorical_options[col] else 0, key=col, help=feature_descriptions.get(col, 'No description available.'))

    # Garage Features
    st.sidebar.subheader('Garage Features')
    col = 'GarageType'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Attchd') if 'Attchd' in categorical_options[col] else 0, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'GarageYrBlt'
    input_data[col] = st.sidebar.number_input(col, min_value=1800.0, value=1980.0, step=1.0, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'GarageFinish'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Unf') if 'Unf' in categorical_options[col] else 0, key=col, help=feature_descriptions.get(col, 'No description available.')) # Use index for 'Unf'
    col = 'GarageCars'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=1, step=1, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'GarageArea'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=300, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'PavedDrive'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Y'), key=col, help=feature_descriptions.get(col, 'No description available.'))


    # Outdoor Features
    st.sidebar.subheader('Outdoor Features')
    col = 'WoodDeckSF'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'OpenPorchSF'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'EnclosedPorch'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = '3SsnPorch'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'ScreenPorch'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'PoolArea'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'PoolQC'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index(np.nan) if np.nan in categorical_options[col] else 0, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'Fence'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index(np.nan) if np.nan in categorical_options[col] else 0, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'MiscFeature'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index(np.nan) if np.nan in categorical_options[col] else 0, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'MiscVal'
    input_data[col] = st.sidebar.number_input(col, min_value=0, value=0, step=10, key=col, help=feature_descriptions.get(col, 'No description available.'))


    # Sale Information
    st.sidebar.subheader('Sale Information')
    col = 'MoSold'
    input_data[col] = st.sidebar.slider(col, 1, 12, 6, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'YrSold'
    input_data[col] = st.sidebar.slider(col, 2006, 2010, 2008, key=col, help=feature_descriptions.get(col, 'No description available.'))
    col = 'SaleType'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('WD'), key=col, help=feature_descriptions.get(col, 'No description available.')) # Use index for 'WD'
    col = 'SaleCondition'
    input_data[col] = st.sidebar.selectbox(col, options=categorical_options[col], index=categorical_options[col].index('Normal'), key=col, help=feature_descriptions.get(col, 'No description available.')) # Use index for 'Normal'


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
        'MoSold', 'YrSold', # Numerical columns used later
        'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'LotFrontage' # Added more numerical columns
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
    unique_ohe_original_cols_expected = []

    if models_loaded and hasattr(onehot_encoder, 'get_feature_names_out'):
        try:
            # Extract original column names from encoder's transformed feature names
            # Example: 'MSZoning__RL' -> 'MSZoning'
            ohe_transformed_feature_names = onehot_encoder.get_feature_names_out()

            # Get unique original column names while preserving the order implied by get_feature_names_out
            for transformed_name in ohe_transformed_feature_names:
                # Split by '__' and take the first part
                original_col = transformed_name.split('__')[0]
                if original_col not in unique_ohe_original_cols_expected:
                    unique_ohe_original_cols_expected.append(original_col)

            # Filter the input DataFrame to keep *only* these identified original columns and ensure they are in the correct order.
            ohe_cols_to_transform = [col for col in unique_ohe_original_cols_expected if col in df_processed.columns]

            # Verify that all expected OHE original columns are present in the input DataFrame
            if len(ohe_cols_to_transform) != len(unique_ohe_original_cols_expected):
                missing_cols = set(unique_ohe_original_cols_expected) - set(ohe_cols_to_transform)
                # st.error(f"Preprocessing error: Input DataFrame is missing expected categorical columns for One-Hot Encoding: {missing_cols}")
                # st.write(f"Expected OHE original columns: {unique_ohe_original_cols_expected}")
                # st.write(f"Available columns in input DF: {df_processed.columns.tolist()}")
                # Instead of failing immediately, try to proceed by dropping missing columns,
                # but log an error, as this indicates a data mismatch.
                st.error(f"Warning: Input DataFrame is missing {len(missing_cols)} columns expected by the One-Hot Encoder: {missing_cols}. These will be skipped.")
                ohe_cols_to_transform = [col for col in unique_ohe_original_cols_expected if col in df_processed.columns] # Re-filter to be safe


            # st.write(f"OHE columns expected by encoder (inferred from get_feature_names_out): {unique_ohe_original_cols_expected}")
            # st.write(f"OHE columns found in input DF for transformation: {ohe_cols_to_transform}")

        except Exception as e:
            st.error(f"Could not determine expected OHE columns from the encoder using get_feature_names_out: {e}. Cannot proceed with OHE.")
            return None # Indicate failure

    else:
        st.error("Could not determine expected OHE columns from the encoder (get_feature_names_out not available). Cannot proceed with OHE.")
        return None # Indicate failure


    if ohe_cols_to_transform:
        # Ensure the columns are in the correct order for transformation
        # Use .reindex to ensure correct order and fill missing columns with NaN if they were unexpectedly absent
        df_processed_ohe_subset = df_processed[ohe_cols_to_transform].copy()

        # st.write(f"Columns being passed to OneHotEncoder: {df_processed_ohe_subset.columns.tolist()}")
        # st.write(f"Data types of columns being passed to OneHotEncoder: {df_processed_ohe_subset.dtypes}")
        # st.write(f"Sample data for OHE columns:\n{df_processed_ohe_subset.head()}")
        # if models_loaded and hasattr(onehot_encoder, 'categories_'):
        #     st.write(f"Encoder's categories structure (length {len(onehot_encoder.categories_)}):")
        #     # Match category index to column name based on ohe_cols_to_transform
        #     for i, cats in enumerate(onehot_encoder.categories_):
        #         col_name = ohe_cols_to_transform[i] if i < len(ohe_cols_to_transform) else f"Unknown_col_{i}"
        #         st.write(f"  Column '{col_name}' (Index {i}): {list(cats)}")


        try:
            # Ensure handle_unknown='ignore' is set for the onehot_encoder
            if hasattr(onehot_encoder, 'handle_unknown'):
                 onehot_encoder.handle_unknown = 'ignore'
                 # st.write("OneHotEncoder handle_unknown is set to 'ignore'.")
            else:
                st.warning("OneHotEncoder does not support handle_unknown='ignore'. Manual handling of unseen categories is crucial.")


            # Convert columns to string type to avoid issues with mixed types before OHE
            for col in ohe_cols_to_transform:
                 if col in df_processed_ohe_subset.columns:
                     # Convert any non-string/non-NaN values to string, handle NaNs
                     df_processed_ohe_subset[col] = df_processed_ohe_subset[col].apply(lambda x: str(x) if pd.notna(x) else np.nan)
                     # Fill remaining NaNs with a placeholder string if 'nan' is not an expected category
                     if df_processed_ohe_subset[col].isnull().any():
                         # Try to get categories for this column from the encoder to see if 'nan' is expected
                         # This requires mapping the current column index to the encoder's categories_ index
                         # which can be tricky if columns were dropped. A safer approach is to
                         # check if 'nan' (as string) is in the encoder's final feature names for this column.

                         # Let's find the index of this original column within the expected OHE columns list
                         try:
                             col_original_index = unique_ohe_original_cols_expected.index(col)
                             # Check if 'nan' is among the categories for this original column in the encoder
                             if col_original_index < len(onehot_encoder.categories_):
                                 encoder_cats_for_col = [str(cat) for cat in onehot_encoder.categories_[col_original_index] if pd.notna(cat)]
                                 if 'nan' not in encoder_cats_for_col:
                                      df_processed_ohe_subset[col] = df_processed_ohe_subset[col].fillna('Missing') # Fill with a placeholder if 'nan' not expected
                                 else:
                                      df_processed_ohe_subset[col] = df_processed_ohe_subset[col].fillna('nan') # Fill with 'nan' string if encoder expects it
                             else:
                                  # Fallback if column index mapping is problematic
                                 df_processed_ohe_subset[col] = df_processed_ohe_subset[col].fillna('Missing') # Default to placeholder


                         except ValueError:
                             # Fallback if column not found in the expected list (shouldn't happen with current logic but for safety)
                             df_processed_ohe_subset[col] = df_processed_ohe_subset[col].fillna('Missing') # Default to placeholder


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
             # Use the original list of columns attempted for OHE transform
             for col in ohe_cols_to_transform:
                 if col in df_processed.columns and not pd.api.types.is_numeric_dtype(df_processed[col]):
                      df_processed = df_processed.drop(columns=col)
             return None # Indicate failure if OHE fails


    # 4. Scale numerical columns (ensure columns exist and handle NaNs)
    # Update numerical columns list to include all columns that should be treated as numerical
    all_potential_numerical_cols = [
        'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
        'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath',
        'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'GarageYrBlt', 'MiscVal',
        'MoSold', 'YrSold', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF',
        'LowQualFinSF', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'EnclosedPorch',
        '3SsnPorch', 'ScreenPorch', 'PoolArea', 'LotFrontage'
    ]
    # Add columns that were label encoded (which are now numeric)
    label_encode_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'Functional', 'GarageFinish']
    all_numeric_cols_for_scaling = list(set(all_potential_numerical_cols + label_encode_cols)) # Combine and get unique

    scale_cols_in_df = [col for col in all_numeric_cols_for_scaling if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]

    if scale_cols_in_df:
        df_processed[scale_cols_in_df] = df_processed[scale_cols_in_df].fillna(0) # Fill NaNs before scaling
        try:
            df_processed[scale_cols_in_df] = loaded_scaler.transform(df_processed[scale_cols_in_df])
        except Exception as e:
             st.warning(f"Could not apply MinMax Scaler: {e}")


    # 5. Align columns with expected features for PCA
    if models_loaded and EXPECTED_FEATURES_AFTER_PREPROCESSING is not None and len(EXPECTED_FEATURES_AFTER_PREPROCESSING) > 0:
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
                         # st.warning(f"Column '{col}' could not be converted to numeric during final alignment. Filling with 0.")
                         df_aligned[col] = 0


        # Ensure all columns are numeric before PCA and fill any remaining NaNs
        for col in df_aligned.columns:
             if not pd.api.types.is_numeric_dtype(df_aligned[col]):
                 try:
                     # Final attempt to convert to numeric and fill NaNs
                     df_aligned[col] = pd.to_numeric(df_aligned[col], errors='coerce')
                     df_aligned[col] = df_aligned[col].fillna(0)
                 except ValueError:
                     # st.warning(f"Column '{col}' is not numeric and could not be converted in final check. It will be dropped for PCA.")
                     df_aligned = df_aligned.drop(columns=col)
             else:
                 df_aligned[col] = df_aligned[col].fillna(0)

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
                 missing_from_expected_pca = set(EXPECTED_FEATURES_AFTER_PREPROCESSING) - set(df_aligned.columns)
                 extra_in_aligned = set(df_aligned.columns) - set(EXPECTED_FEATURES_AFTER_PREPROCESSING)
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
st.write("Adjust the house characteristics in the sidebar to get a price prediction.")


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
                    # Format prediction as currency with commas
                    st.success(f"The predicted value of the house is: ${predictions[0]:,.2f}")

            else:
                 st.warning("The loaded model does not have the 'n_features_in_' attribute. Could not verify feature consistency. Attempting to predict anyway...")
                 try:
                     predictions = loaded_model.predict(processed_df)
                     st.subheader("Predicted House Value:")
                     # Format prediction as currency with commas
                     st.success(f"The predicted value of the house is: ${predictions[0]:,.2f}")

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
