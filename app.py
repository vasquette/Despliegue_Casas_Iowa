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
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('PoolArea', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'PoolQC'
    # Found and removed the duplicate PoolQC definition here
    # widget_keys.add(widget_key) # Don't add key if removing the definition
    # input_data[widget_key] = st.sidebar.selectbox('PoolQC', options=[np.nan, 'Ex', 'Gd', 'TA', 'Fa'], index=1, key=widget_key)

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


    # Widgets para variables categóricas (ejemplos con opciones comunes)
    widget_key = 'MSZoning'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('MSZoning', options=['RL', 'RM', 'C (all)', 'FV', 'RH'], index=0, key=widget_key)

    widget_key = 'Street'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Street', options=['Pave', 'Grvl'], index=0, key=widget_key)

    widget_key = 'Alley'
    widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Alley', options=[np.nan, 'Grvl', 'Pave'], index=1)
