import streamlit as st
import pandas as pd
import joblib
import numpy as np
import scipy.sparse # Importar scipy.sparse

# --- Cargar Modelos y Preprocesadores ---
# Intenta cargar cada archivo y maneja los errores si no se encuentran
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
    st.error(f"Error al cargar archivo: {e}. Asegúrate de que todos los archivos .pkl estén en el directorio correcto.")
    models_loaded = False
except Exception as e:
    st.error(f"Ocurrió un error al cargar modelos o preprocesadores: {e}")
    models_loaded = False


# --- Función para Generar Datos por Defecto (80 variables) y Widgets de Entrada ---
def get_input_data_from_widgets():
    """Genera widgets de entrada para cada variable y retorna un DataFrame."""
    st.sidebar.header('Ingresa las Características de la Casa')

    # Definir valores por defecto o rangos razonables para los widgets
    # Estos son solo ejemplos, deberías ajustarlos según tu análisis exploratorio
    input_data = {}

    # Use a set to track keys and identify duplicates within this function
    # NOTE: This set should ideally track keys globally if widgets were defined outside this function
    # However, based on current code structure, widgets are only in this function.
    # A more robust check would require analyzing the Streamlit API usage pattern.
    # For this specific code, checking within the function is sufficient.
    widget_keys = set()

    # Widgets para variables numéricas (ejemplos)
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
    # Found and removed the duplicate PoolArea definition here
    # widget_keys.add(widget_key) # Don't add key if removing the definition
    # input_data[widget_key] = st.sidebar.number_input('PoolArea', min_value=0, value=0, step=10, key=widget_key)

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
    input_data['Id'] = 0 # Un ID placeholder


    # Widgets para variables categóricas (ejemplos con opciones comunes)
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
    input_data[widget_key] = st.sidebar.selectbox('BsmtQual', options=['Gd', 'TA', 'Ex', 'Fa', np.nan], index=1, key=widget_key) # Usar index=1 para 'TA' como defecto

    widget_key = 'BsmtCond'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtCond', options=['TA', 'Gd', 'Fa', 'Po', np.nan], index=0, key=widget_key) # Usar index=0 para 'TA' como defecto

    widget_key = 'BsmtExposure'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtExposure', options=['No', 'Gd', 'Mn', 'Av', np.nan], index=0, key=widget_key)

    widget_key = 'BsmtFinType1'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtFinType1', options=['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ', np.nan], index=1, key=widget_key) # Usar index=1 para 'ALQ' como defecto

    widget_key = 'BsmtFinType2'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtFinType2', options=['Unf', 'Rec', 'LwQ', 'GLQ', 'ALQ', 'BLQ', np.nan], index=0, key=widget_key) # Usar index=0 para 'Unf' como defecto

    widget_key = 'Heating'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Heating', options=['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'], index=0, key=widget_key)

    widget_key = 'HeatingQC'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('HeatingQC', options=['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=0, key=widget_key) # Usar index=0 para 'Ex' como defecto

    widget_key = 'CentralAir'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('CentralAir', options=['Y', 'N'], index=0, key=widget_key)

    widget_key = 'Electrical'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Electrical', options=['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', np.nan], index=1, key=widget_key)

    widget_key = 'KitchenQual'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('KitchenQual', options=['Gd', 'TA', 'Ex', 'Fa'], index=1, key=widget_key) # Usar index=1 para 'TA' como defecto

    widget_key = 'Functional'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Functional', options=['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev'], index=0, key=widget_key) # Usar index=0 para 'Typ' como defecto

    widget_key = 'FireplaceQu'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('FireplaceQu', options=[np.nan, 'Gd', 'TA', 'Fa', 'Ex', 'Po'], index=1, key=widget_key)

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
    input_data['Id'] = 0 # Un ID placeholder


    # Convertir el diccionario a DataFrame
    return pd.DataFrame([input_data])


# --- Función para Preprocesar los Datos de Entrada ---
def preprocess_input(input_df):
    """Aplica los mismos pasos de preprocesamiento que se usaron en el entrenamiento."""
    df_processed = input_df.copy()

    # Asegurarse de que las columnas esperadas existan en el DataFrame de entrada
    # Si alguna columna esperada falta, añadirla con NaN o un valor por defecto adecuado
    expected_original_cols = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'BsmtFullBath', 'FullBath', 'HalfBath', 'Fireplaces', 'GarageCars', 'WoodDeckSF', 'OpenPorchSF', 'MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'Exterior1st', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition']
    for col in expected_original_cols:
        if col not in df_processed.columns:
            # Add missing column with NaN or a suitable default
            df_processed[col] = np.nan # Or a suitable default based on column type
            #st.warning(f"Columna '{col}' faltante en los datos de entrada. Añadida con NaN.") # Disabled to avoid too many warnings in the UI


    # Select only the columns that the model expects (those used in training)
    # Note: With the widgets, we should be generating all expected columns.
    # This selection might be redundant if the widgets cover all necessary columns.
    # However, we keep it for safety if the input comes from another source or there is a mismatch.
    # Ensure all selected columns exist in df_processed
    cols_to_select = [col for col in expected_original_cols if col in df_processed.columns]
    df_processed = df_processed[cols_to_select].copy()


    # 1. Convert to category if necessary (ensure columns exist)
    # Identify categorical columns in the current DataFrame
    current_categorical_cols = df_processed.select_dtypes(include='object').columns
    for col in current_categorical_cols:
         if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')


    # 2. Apply LabelEncoder (ensure handling of unseen labels)
    # We remove 'OverallQual' and 'OverallCond' from this list as they are numeric
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
                 # Handle unseen labels: replace with a label the encoder knows.
                 # A common strategy is to use the most frequent value or a placeholder.
                 # Here, we will try to convert to category, and unseen values will become NaN by default.
                 # Then, if necessary, we will fill those NaNs with a numerical value.

                 # Ensure column categories match those of the encoder
                 # This might throw an error if there are unseen labels that are not handled.
                 # A safe way is to convert to CategoricalDtype with the encoder's categories,
                 # which will convert unseen ones to NaN.
                 df_processed[col] = df_processed[col].astype(pd.CategoricalDtype(categories=encoder.classes_))

                 # Apply transformation (this will convert categories to their numerical codes)
                 df_processed[col] = df_processed[col].cat.codes
                 # Values that were NaN (unseen labels) will become -1 by .cat.codes
                 # We can fill these -1 with a default numerical value if necessary
                 df_processed[col] = df_processed[col].replace(-1, np.nan) # Convert -1 back to NaN
                 # Fill NaN with a default numerical value if necessary (ej. media o moda del entrenamiento, o 0)
                 df_processed[col] = df_processed[col].fillna(0) # Fill NaN with 0 as example

             except Exception as e:
                 st.warning(f"Could not apply LabelEncoder to column '{col}': {e}")
                 # If it fails, try converting the column to numeric and filling NaN
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(0) # Fill with 0 if conversion fails or there are NaNs


    # Convert LabelEncoder columns (now numeric) to numeric type if not already
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
            # Fill NaN values in categorical columns with a placeholder
            for col in ohe_cols_in_df:
                 df_processed[col] = df_processed[col].fillna('Missing') # Fill NaN with 'Missing' string

            # Convert to string type to ensure consistency before OHE
            df_processed[ohe_cols_in_df] = df_processed[ohe_cols_in_df].astype(str)

            # Applying One-Hot Encoding with handle_unknown='ignore' if possible
            try:
                # Check if the encoder supports handle_unknown
                if hasattr(onehot_encoder, 'handle_unknown'):
                     encoded_data = onehot_encoder.transform(df_processed[ohe_cols_in_df])
                else:
                     # If loaded encoder doesn't have handle_unknown, try re-initializing
                     st.warning("Loaded One-Hot Encoder does not support 'handle_unknown'. Attempting re-initialization.")
                     from sklearn.preprocessing import OneHotEncoder
                     # Use the categories the original encoder knows
                     known_categories = onehot_encoder.categories_
                     # Re-create an encoder with known categories and handle_unknown='ignore'
                     temp_ohe = OneHotEncoder(categories=known_categories, handle_unknown='ignore', sparse_output=onehot_encoder.sparse_output)
                     encoded_data = temp_ohe.transform(df_processed[ohe_cols_in_df])
                     st.success("Successfully re-initialized and applied One-Hot Encoder with handle_unknown='ignore'.")
                     onehot_encoder = temp_ohe # Replace the old encoder with the new one


            except ValueError as e:
                 st.error(f"One-Hot Encoding failed due to value error: {e}. This might still be due to unknown categories.")
                 encoded_data = None # Indicate failure
            except Exception as e:
                 st.error(f"An unexpected error occurred during One-Hot Encoding: {e}")
                 encoded_data = None # Indicate failure


            if encoded_data is not None:
                # Convert the result to a dense array if it is a sparse matrix
                if isinstance(encoded_data, scipy.sparse.csr.csr_matrix):
                     encoded_data = encoded_data.toarray()


                # Create a DataFrame with the encoded columns
                # Use get_feature_names_out with the input DataFrame columns to avoid mismatch errors
                # If handle_unknown='ignore' was used, get_feature_names_out might not include columns for ignored categories.
                # Use the feature names from the original encoder if possible.
                try:
                     encoded_column_names = onehot_encoder.get_feature_names_out(ohe_cols_in_df)
                except AttributeError:
                     # Fallback if get_feature_names_out is not available (older sklearn versions)
                     st.warning("Using feature names from the current encoder (may not be perfectly aligned).")
                     encoded_column_names = onehot_encoder.get_feature_names_out(ohe_cols_in_df)


                encoded_df = pd.DataFrame(encoded_data, columns=encoded_column_names, index=df_processed.index)

                # Drop the original categorical columns
                df_processed = df_processed.drop(columns=ohe_cols_in_df)

                # Concatenate the original DataFrame with the encoded columns
                df_processed = pd.concat([df_processed, encoded_df], axis=1)
            else:
                 # If OHE failed, drop the original categorical columns to proceed with PCA/prediction
                 st.warning("One-Hot Encoding failed. Dropping original categorical columns.")
                 df_processed = df_processed.drop(columns=ohe_cols_in_df)


        except Exception as e:
             st.warning(f"An error occurred during One-Hot Encoding processing: {e}")
             # If a critical error occurred, attempt to drop original categorical columns to prevent further issues
             for col in ohe_cols_in_df:
                 if col in df_processed.columns and not pd.api.types.is_numeric_dtype(df_processed[col]):
                      df_processed = df_processed.drop(columns=col)


    # 4. Scale numeric columns (ensure columns exist and handle NaN values after OHE if applicable)
    numerical_cols_to_scale = ['OverallQual', 'OverallCond','LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF']
    scale_cols_in_df = [col for col in numerical_cols_to_scale if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]

    if scale_cols_in_df:
        # Fill possible NaNs with the mean or a default value before scaling if necessary
        # Here we use 0, but it's better to use the mean from the training set
        df_processed[scale_cols_in_df] = df_processed[scale_cols_in_df].fillna(0)
        try:
            df_processed[scale_cols_in_df] = loaded_scaler.transform(df_processed[scale_cols_in_df])
        except Exception as e:
             st.warning(f"Could not apply MinMax Scaler: {e}")


    # 5. Align columns with those expected by the PCA model (handle missing/extra columns)
    if hasattr(pca_model, 'feature_names_in_'):
        expected_features = pca_model.feature_names_in_
        # Reindex the processed DataFrame to match the columns expected by PCA
        # Fill missing values with 0, if this is appropriate.
        df_aligned = df_processed.reindex(columns=expected_features, fill_value=0)

        # Ensure all columns are numeric before PCA
        for col in df_aligned.columns:
             if not pd.api.types.is_numeric_dtype(df_aligned[col]):
                 try:
                     df_aligned[col] = pd.to_numeric(df_aligned[col])
                     df_aligned[col] = df_aligned[col].fillna(0) # Fill NaN after conversion if any
                 except ValueError:
                     st.warning(f"Column '{col}' is not numeric and could not be converted. It will be dropped for PCA.")
                     df_aligned = df_aligned.drop(columns=col)


        # 6. Apply PCA
        # Check if the number of features in df_aligned matches what's expected by PCA
        if hasattr(pca_model, 'n_features_in_') and df_aligned.shape[1] != pca_model.n_features_in_:
             st.error(f"The number of features after preprocessing ({df_aligned.shape[1]}) does not match what's expected by the PCA model ({pca_model.n_features_in_}). PCA will not be applied.")
             # If columns don't match, we cannot apply PCA. Return the aligned DataFrame without PCA.
             df_pca = df_aligned.copy()
        else:
            try:
                df_pca = pca_model.transform(df_aligned)
                 # Convert PCA result to DataFrame for consistency
                pca_column_names = [f'principal_component_{i+1}' for i in range(df_pca.shape[1])]
                df_pca = pd.DataFrame(df_pca, columns=pca_column_names, index=df_aligned.index)

            except Exception as e:
                st.warning(f"Could not apply PCA: {e}")
                # If PCA fails, return the aligned DataFrame without PCA or handle the error.
                # We will return the aligned DataFrame, but prediction might fail.
                df_pca = df_aligned.copy() # If PCA fails, use the aligned DataFrame

    else:
        st.warning("The loaded PCA model does not have 'feature_names_in_'. Could not align the DataFrame.")
        df_pca = df_processed.copy() # If unable to align for PCA, use the processed DataFrame (without PCA)


    return df_pca


# --- Streamlit User Interface ---
st.title("Predicción del Valor de Casas en Iowa")

st.write("Esta aplicación predice el valor de una casa basado en sus características. Ingresa los valores de las características a continuación para obtener una predicción.")

# Use the function that generates widgets to get input data
input_df = get_input_data_from_widgets()


# --- Make Predictions ---
if input_df is not None and models_loaded:
    st.write("Preprocessing data and making predictions...")

    try:
        processed_df = preprocess_input(input_df)

        # Make predictions if the model is loaded and preprocessing was successful
        if 'loaded_model' in locals() and processed_df is not None:
            # Ensure the number of features after preprocessing/PCA
            # matches what's expected by the loaded model
            if hasattr(loaded_model, 'n_features_in_'):
                if processed_df.shape[1] != loaded_model.n_features_in_:
                    st.error(f"The number of features after preprocessing ({processed_df.shape[1]}) does not match what's expected by the model ({loaded_model.n_features_in_}). Could not make prediction.")
                    st.write("Expected features by the model:", loaded_model.feature_names_in_)
                    st.write("Features after preprocessing:", processed_df.columns.tolist())

                else:
                    predictions = loaded_model.predict(processed_df)
                    st.subheader("Predicción del Valor de la Casa:")
                    # Display prediction prominently
                    st.success(f"El valor predicho de la casa es: ${predictions[0]:,.2f}") # Currency format

            else:
                 st.warning("The loaded model does not have the 'n_features_in_' attribute. Could not verify feature consistency. Attempting to predict anyway...")
                 # Attempt to predict anyway, but with caution
                 try:
                     predictions = loaded_model.predict(processed_df)
                     st.subheader("Predicción del Valor de la Casa:")
                     st.success(f"El valor predicho de la casa es: ${predictions[0]:,.2f}")

                 except Exception as e:
                     st.error(f"Error making predictions: {e}")


        else:
            st.warning("Could not load the model or preprocess data to make predictions.")

    except Exception as e:
        st.error(f"An error occurred during preprocessing or prediction: {e}")

elif not models_loaded:
    st.warning("Models and preprocessors were not loaded correctly. Cannot make predictions.")

# Display the input data generated by widgets (optional, for debugging)
# st.sidebar.subheader("Generated Input Data:")
# st.sidebar.dataframe(input_df)
