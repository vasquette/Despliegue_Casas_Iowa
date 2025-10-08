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

    # Check for duplicate keys within this function before adding to input_data
    widget_keys = set()

    # Widgets para variables numéricas (ejemplos)
    widget_key = 'MSSubClass'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('MSSubClass', options=[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190], index=0, key=widget_key)

    widget_key = 'LotFrontage'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('LotFrontage', min_value=0.0, value=70.0, step=1.0, key=widget_key)

    widget_key = 'LotArea'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('LotArea', min_value=0, value=10500, step=100, key=widget_key)

    widget_key = 'OverallQual'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.slider('OverallQual', 1, 10, 5, key=widget_key)

    widget_key = 'OverallCond'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.slider('OverallCond', 1, 9, 5, key=widget_key)

    widget_key = 'YearBuilt'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('YearBuilt', min_value=1800, value=1980, step=1, key=widget_key)

    widget_key = 'YearRemodAdd'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('YearRemodAdd', min_value=1800, value=1980, step=1, key=widget_key)

    widget_key = 'MasVnrArea'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('MasVnrArea', min_value=0.0, value=0.0, step=10.0, key=widget_key)

    widget_key = 'BsmtFinSF1'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BsmtFinSF1', min_value=0, value=500, step=10, key=widget_key)

    widget_key = 'BsmtFinSF2'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BsmtFinSF2', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'BsmtUnfSF'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BsmtUnfSF', min_value=0, value=500, step=10, key=widget_key)

    widget_key = 'TotalBsmtSF'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('TotalBsmtSF', min_value=0, value=1000, step=10, key=widget_key)

    widget_key = '1stFlrSF'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('1stFlrSF', min_value=0, value=1000, step=10, key=widget_key)

    widget_key = '2ndFlrSF'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('2ndFlrSF', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'LowQualFinSF'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('LowQualFinSF', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'GrLivArea'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('GrLivArea', min_value=0, value=1000, step=10, key=widget_key)

    widget_key = 'BsmtFullBath'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BsmtFullBath', min_value=0, value=0, step=1, key=widget_key)

    widget_key = 'BsmtHalfBath'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BsmtHalfBath', min_value=0, value=0, step=1, key=widget_key)

    widget_key = 'FullBath'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('FullBath', min_value=0, value=1, step=1, key=widget_key)

    widget_key = 'HalfBath'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('HalfBath', min_value=0, value=0, step=1, key=widget_key)

    widget_key = 'BedroomAbvGr'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('BedroomAbvGr', min_value=0, value=3, step=1, key=widget_key)

    widget_key = 'KitchenAbvGr'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('KitchenAbvGr', min_value=0, value=1, step=1, key=widget_key)

    widget_key = 'TotRmsAbvGrd'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('TotRmsAbvGrd', min_value=0, value=5, step=1, key=widget_key)

    widget_key = 'Fireplaces'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('Fireplaces', min_value=0, value=0, step=1, key=widget_key)

    widget_key = 'GarageType'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('GarageType', options=['Attchd', 'Detchd', 'BuiltIn', 'CarPort', np.nan, 'Basment', '2Types'], index=0, key=widget_key)

    widget_key = 'GarageYrBlt'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('GarageYrBlt', min_value=1800.0, value=1980.0, step=1.0, key=widget_key)

    widget_key = 'GarageCars'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('GarageCars', min_value=0, value=1, step=1, key=widget_key)

    widget_key = 'GarageArea'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('GarageArea', min_value=0, value=300, step=10, key=widget_key)

    widget_key = 'WoodDeckSF'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('WoodDeckSF', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'OpenPorchSF'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('OpenPorchSF', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'EnclosedPorch'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('EnclosedPorch', min_value=0, value=0, step=10, key=widget_key)

    widget_key = '3SsnPorch'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('3SsnPorch', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'ScreenPorch'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('ScreenPorch', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'PoolArea'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('PoolArea', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'PoolQC'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('PoolQC', options=[np.nan, 'Ex', 'Gd', 'TA', 'Fa'], index=1, key=widget_key)

    widget_key = 'Fence'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Fence', options=[np.nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'], index=1, key=widget_key)

    widget_key = 'MiscFeature'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('MiscFeature', options=[np.nan, 'Shed', 'Gar2', 'Othr', 'TenC'], index=1, key=widget_key)

    widget_key = 'MiscVal'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('MiscVal', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'MoSold'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.slider('MoSold', 1, 12, 6, key=widget_key)

    widget_key = 'YrSold'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.slider('YrSold', 2006, 2010, 2008, key=widget_key)

    # Widgets para variables categóricas (ejemplos con opciones comunes)
    widget_key = 'MSZoning'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('MSZoning', options=['RL', 'RM', 'C (all)', 'FV', 'RH'], index=0, key=widget_key)

    widget_key = 'Street'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Street', options=['Pave', 'Grvl'], index=0, key=widget_key)

    widget_key = 'Alley'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Alley', options=[np.nan, 'Grvl', 'Pave'], index=1, key=widget_key)

    widget_key = 'LotShape'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('LotShape', options=['Reg', 'IR1', 'IR2', 'IR3'], index=0, key=widget_key)

    widget_key = 'LandContour'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('LandContour', options=['Lvl', 'Bnk', 'HLS', 'Low'], index=0, key=widget_key)

    widget_key = 'Utilities'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Utilities', options=['AllPub', 'NoSeWa'], index=0, key=widget_key)

    widget_key = 'LotConfig'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('LotConfig', options=['Inside', 'Frontage', 'Corner', 'CulDSac', 'FR2', 'FR3'], index=0, key=widget_key)

    widget_key = 'LandSlope'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('LandSlope', options=['Gtl', 'Mod', 'Sev'], index=0, key=widget_key)

    widget_key = 'Neighborhood'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Neighborhood', options=['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber', 'IDOTC', 'ClearCr', 'StoneBr', 'SWISU', 'Blmngtn', 'MeadowV', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'], index=0, key=widget_key)

    widget_key = 'Condition1'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Condition1', options=['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe'], index=0, key=widget_key)

    widget_key = 'Condition2'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Condition2', options=['Norm', 'Artery', 'Feedr', 'RRNn', 'RRAn', 'RRAe', 'PosN', 'PosA'], index=0, key=widget_key)

    widget_key = 'BldgType'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BldgType', options=['1Fam', '2FmCon', 'Duplex', 'TwnhsE', 'Twnhs'], index=0, key=widget_key)

    widget_key = 'HouseStyle'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('HouseStyle', options=['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'], index=0, key=widget_key)

    widget_key = 'RoofStyle'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('RoofStyle', options=['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed'], index=0, key=widget_key)

    widget_key = 'RoofMatl'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('RoofMatl', options=['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Roll', 'ClyTile'], index=0, key=widget_key)

    widget_key = 'Exterior1st'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Exterior1st', options=['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'CBlock', 'BrkComm', 'AsphShn', 'VinylSd', 'ImStucc', 'Stone', 'Other'], index=0, key=widget_key)

    widget_key = 'Exterior2nd'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Exterior2nd', options=['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'BrkFace', 'Wd Sdng', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'CBlock', 'BrkComm', 'AsphShn', 'VinylSd', 'ImStucc', 'Stone', 'Other'], index=0, key=widget_key)

    widget_key = 'MasVnrType'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('MasVnrType', options=['None', 'BrkFace', 'Stone', 'BrkCmn', np.nan], index=1, key=widget_key)

    widget_key = 'Foundation'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Foundation', options=['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'], index=0, key=widget_key)

    widget_key = 'BsmtQual'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtQual', options=['Gd', 'TA', 'Ex', 'Fa', np.nan], index=1, key=widget_key) # Usar index=1 para 'TA' como defecto

    widget_key = 'BsmtCond'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtCond', options=['TA', 'Gd', 'Fa', 'Po', np.nan], index=0, key=widget_key) # Usar index=0 para 'TA' como defecto

    widget_key = 'BsmtExposure'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtExposure', options=['No', 'Gd', 'Mn', 'Av', np.nan], index=0, key=widget_key)

    widget_key = 'BsmtFinType1'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtFinType1', options=['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ', np.nan], index=1, key=widget_key) # Usar index=1 para 'ALQ' como defecto

    widget_key = 'BsmtFinType2'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('BsmtFinType2', options=['Unf', 'Rec', 'LwQ', 'GLQ', 'ALQ', 'BLQ', np.nan], index=0, key=widget_key) # Usar index=0 para 'Unf' como defecto

    widget_key = 'Heating'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Heating', options=['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'], index=0, key=widget_key)

    widget_key = 'HeatingQC'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('HeatingQC', options=['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=0, key=widget_key) # Usar index=0 para 'Ex' como defecto

    widget_key = 'CentralAir'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('CentralAir', options=['Y', 'N'], index=0, key=widget_key)

    widget_key = 'Electrical'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Electrical', options=['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', np.nan], index=1, key=widget_key)

    widget_key = 'KitchenQual'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('KitchenQual', options=['Gd', 'TA', 'Ex', 'Fa'], index=1, key=widget_key) # Usar index=1 para 'TA' como defecto

    widget_key = 'Functional'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Functional', options=['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev'], index=0, key=widget_key) # Usar index=0 para 'Typ' como defecto

    widget_key = 'FireplaceQu'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('FireplaceQu', options=[np.nan, 'Gd', 'TA', 'Fa', 'Ex', 'Po'], index=1, key=widget_key)

    widget_key = 'PoolArea'
    # Check if PoolArea key is already used. If so, this is a duplicate.
    if widget_key in widget_keys:
        st.warning(f"Duplicate key found: {widget_key}. Removing the duplicate widget definition.")
        # Do not add to input_data again if it's a duplicate we are removing
    else:
        widget_keys.add(widget_key)
        input_data[widget_key] = st.sidebar.number_input('PoolArea', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'PoolQC'
    # Check if PoolQC key is already used. If so, this is a duplicate.
    if widget_key in widget_keys:
        st.warning(f"Duplicate key found: {widget_key}. Removing the duplicate widget definition.")
        # Do not add to input_data again if it's a duplicate we are removing
    else:
        widget_keys.add(widget_key)
        input_data[widget_key] = st.sidebar.selectbox('PoolQC', options=[np.nan, 'Ex', 'Gd', 'TA', 'Fa'], index=1, key=widget_key)

    widget_key = 'Fence'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('Fence', options=[np.nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'], index=1, key=widget_key)

    widget_key = 'MiscFeature'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.selectbox('MiscFeature', options=[np.nan, 'Shed', 'Gar2', 'Othr', 'TenC'], index=1, key=widget_key)

    widget_key = 'MiscVal'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.number_input('MiscVal', min_value=0, value=0, step=10, key=widget_key)

    widget_key = 'MoSold'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.slider('MoSold', 1, 12, 6, key=widget_key)

    widget_key = 'YrSold'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = st.sidebar.slider('YrSold', 2006, 2010, 2008, key=widget_key)

    widget_key = 'Id'
    if widget_key in widget_keys: st.warning(f"Duplicate key found: {widget_key}")
    else: widget_keys.add(widget_key)
    input_data[widget_key] = 0 # Un ID placeholder - note: not a Streamlit widget, so key isn't needed here, but keeping for consistency

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
            # Añadir la columna faltante con NaN o un valor por defecto
            df_processed[col] = np.nan # O un valor por defecto adecuado al tipo de columna
            #st.warning(f"Columna '{col}' faltante en los datos de entrada. Añadida con NaN.") # Desactivado para evitar demasiados warnings en la UI


    # Seleccionar solo las columnas que el modelo espera (las que se usaron en el entrenamiento)
    # Nota: Con los widgets, deberíamos estar generando todas las columnas esperadas.
    # Esta selección puede ser redundante si los widgets cubren todas las columnas necesarias.
    # Sin embargo, la mantenemos por seguridad si el input viene de otra fuente o si hay un desajuste.
    # Asegurarse de que todas las columnas seleccionadas existan en df_processed
    cols_to_select = [col for col in expected_original_cols if col in df_processed.columns]
    df_processed = df_processed[cols_to_select].copy()


    # 1. Convertir a categoría si es necesario (asegúrate de que las columnas existan)
    # Identificar columnas categóricas en el DataFrame actual
    current_categorical_cols = df_processed.select_dtypes(include='object').columns
    for col in current_categorical_cols:
         if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')


    # 2. Aplicar LabelEncoder (asegúrate de manejar etiquetas no vistas)
    # Eliminamos 'OverallQual' y 'OverallCond' de esta lista ya que son numéricas
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
                 # Manejar etiquetas no vistas: reemplazar con una etiqueta que el encoder conozca.
                 # Una estrategia común es usar el valor más frecuente o un placeholder.
                 # Aquí, intentaremos convertir a la categoría y los valores no vistos se convertirán a NaN por defecto.
                 # Luego, si es necesario, rellenaremos esos NaN con un valor numérico.

                 # Asegurarse de que las categorías de la columna coincidan con las del encoder
                 # Esto puede lanzar un error si hay etiquetas no vistas que no se manejan.
                 # Una forma segura es convertir a CategoricalDtype con las categorías del encoder,
                 # lo que convertirá las no vistas a NaN.
                 df_processed[col] = df_processed[col].astype(pd.CategoricalDtype(categories=encoder.classes_))

                 # Aplicar la transformación (esto convertirá las categorías a sus códigos numéricos)
                 df_processed[col] = df_processed[col].cat.codes
                 # Los valores que eran NaN (etiquetas no vistas) se convertirán a -1 por .cat.codes
                 # Podemos rellenar estos -1 con un valor por defecto numérico si es necesario
                 df_processed[col] = df_processed[col].replace(-1, np.nan) # Convertir -1 de vuelta a NaN
                 # Rellenar NaN con un valor por defecto numérico si es necesario (ej. media o moda del entrenamiento, o 0)
                 df_processed[col] = df_processed[col].fillna(0) # Rellenar NaN con 0 como ejemplo

             except Exception as e:
                 st.warning(f"No se pudo aplicar LabelEncoder a la columna '{col}': {e}")
                 # Si falla, intentar convertir la columna a numérica y rellenar NaN
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(0) # Rellenar con 0 si la conversión falla o hay NaN


    # Convertir columnas de LabelEncoder (ahora numéricas) a tipo numérico si no lo son ya
    for col in label_encode_cols:
        if col in df_processed.columns:
             if not pd.api.types.is_numeric_dtype(df_processed[col]):
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(0)


    # 3. Aplicar One-Hot Encoding (asegúrate de que las columnas existan y manejar columnas faltantes/extra)
    categorical_cols_ohe = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'Exterior1st', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition']
    ohe_cols_in_df = [col for col in categorical_cols_ohe if col in df_processed.columns]

    if ohe_cols_in_df:
        try:
            # Llenar valores NaN en columnas categóricas con un marcador de posición
            for col in ohe_cols_in_df:
                 df_processed[col] = df_processed[col].fillna('Missing').astype(str) # Llenar NaN y asegurar tipo string


            # Aplicar One-Hot Encoding
            # Usar handle_unknown='ignore' si el encoder fue entrenado con esta opción
            # Si no, asegurarse de que el input solo tenga categorías vistas por el encoder
            # Si el encoder no soporta handle_unknown='ignore' y hay categorías no vistas, esto fallará.
            # Asumimos que el onehot_encoder cargado soporta handle_unknown='ignore'
            encoded_data = onehot_encoder.transform(df_processed[ohe_cols_in_df])

            # Convertir el resultado a array denso si es una matriz dispersa
            if isinstance(encoded_data, scipy.sparse.csr.csr_matrix):
                 encoded_data = encoded_data.toarray()


            # Crear un DataFrame con las columnas codificadas
            # Usar get_feature_names_out con las columnas del DataFrame de entrada para evitar errores de coincidencia
            encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(ohe_cols_in_df), index=df_processed.index)

            # Eliminar las columnas categóricas originales
            df_processed = df_processed.drop(columns=ohe_cols_in_df)

            # Concatenar el DataFrame original con las columnas codificadas
            df_processed = pd.concat([df_processed, encoded_df], axis=1)
        except Exception as e:
             st.warning(f"No se pudo aplicar One-Hot Encoding: {e}")
             # Si falla OHE, las columnas originales categóricas quedan.
             # Para evitar errores en PCA, podríamos intentar eliminarlas o convertirlas a un tipo numérico por defecto.
             for col in ohe_cols_in_df:
                 if col in df_processed.columns and not pd.api.types.is_numeric_dtype(df_processed[col]):
                      df_processed = df_processed.drop(columns=col)


    # 4. Escalar columnas numéricas (asegúrate de que las columnas existan y manejar valores NaN después de OHE si aplica)
    numerical_cols_to_scale = ['OverallQual', 'OverallCond','LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF']
    scale_cols_in_df = [col for col in numerical_cols_to_scale if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]

    if scale_cols_in_df:
        # Rellenar posibles NaN con la media o un valor por defecto antes de escalar si es necesario
        # Aquí usamos 0, pero es mejor usar la media del conjunto de entrenamiento
        df_processed[scale_cols_in_df] = df_processed[scale_cols_in_df].fillna(0)
        try:
            df_processed[scale_cols_in_df] = loaded_scaler.transform(df_processed[scale_cols_in_df])
        except Exception as e:
             st.warning(f"No se pudo aplicar MinMax Scaler: {e}")


    # 5. Alinear columnas con las esperadas por el modelo PCA (manejar columnas faltantes/extra)
    if hasattr(pca_model, 'feature_names_in_'):
        expected_features = pca_model.feature_names_in_
        # Reindexar el DataFrame procesado para que coincida con las columnas esperadas por el PCA
        # Llenar con 0 los valores faltantes, si esto es apropiado.
        df_aligned = df_processed.reindex(columns=expected_features, fill_value=0)

        # Asegurarse de que todas las columnas sean numéricas antes de PCA
        for col in df_aligned.columns:
             if not pd.api.types.is_numeric_dtype(df_aligned[col]):
                 try:
                     df_aligned[col] = pd.to_numeric(df_aligned[col])
                     df_aligned[col] = df_aligned[col].fillna(0) # Rellenar NaN después de la conversión si los hay
                 except ValueError:
                     st.warning(f"La columna '{col}' no es numérica y no se pudo convertir. Se eliminará para PCA.")
                     df_aligned = df_aligned.drop(columns=col)


        # 6. Aplicar PCA
        # Verificar si el número de características en df_aligned coincide con el esperado por el PCA
        if hasattr(pca_model, 'n_features_in_') and df_aligned.shape[1] != pca_model.n_features_in_:
             st.error(f"El número de características después del preprocesamiento ({df_aligned.shape[1]}) no coincide con las esperadas por el modelo PCA ({pca_model.n_features_in_}). No se aplicará PCA.")
             # Si las columnas no coinciden, no podemos aplicar PCA. Retornar el DataFrame alineado sin PCA.
             df_pca = df_aligned.copy()
        else:
            try:
                df_pca = pca_model.transform(df_aligned)
                 # Convertir el resultado de PCA a DataFrame para consistencia
                pca_column_names = [f'principal_component_{i+1}' for i in range(df_pca.shape[1])]
                df_pca = pd.DataFrame(df_pca, columns=pca_column_names, index=df_aligned.index)

            except Exception as e:
                st.warning(f"No se pudo aplicar PCA: {e}")
                # Si falla PCA, retornar el DataFrame alineado sin PCA o manejar el error.
                # Retornaremos el DataFrame alineado, pero la predicción podría fallar.
                df_pca = df_aligned.copy() # Si PCA falla, usamos el DataFrame alineado

    else:
        st.warning("El modelo PCA no tiene 'feature_names_in_'. No se pudo alinear el DataFrame.")
        df_pca = df_processed.copy() # Si no se puede alinear para PCA, usamos el DataFrame preprocesado (sin PCA)


    return df_pca


# --- Interfaz de Usuario con Streamlit ---
st.title("Predicción del Valor de Casas en Iowa")

st.write("Esta aplicación predice el valor de una casa basado en sus características. Ingresa los valores de las características a continuación para obtener una predicción.")

# Usar la función que genera widgets para obtener los datos de entrada
input_df = get_input_data_from_widgets()


# --- Realizar Predicciones ---
if input_df is not None and models_loaded:
    st.write("Preprocesando datos y realizando predicciones...")

    try:
        processed_df = preprocess_input(input_df)

        # Realizar predicciones si el modelo está cargado y el preprocesamiento fue exitoso
        if 'loaded_model' in locals() and processed_df is not None:
            # Asegurarse de que el número de características después del preprocesamiento/PCA
            # coincida con lo esperado por el modelo cargado
            if hasattr(loaded_model, 'n_features_in_'):
                if processed_df.shape[1] != loaded_model.n_features_in_:
                    st.error(f"El número de características después del preprocesamiento ({processed_df.shape[1]}) no coincide con las esperadas por el modelo ({loaded_model.n_features_in_}). No se pudo realizar la predicción.")
                    st.write("Características esperadas por el modelo:", loaded_model.feature_names_in_)
                    st.write("Características después del preprocesamiento:", processed_df.columns.tolist())

                else:
                    predictions = loaded_model.predict(processed_df)
                    st.subheader("Predicción del Valor de la Casa:")
                    # Mostrar la predicción de manera destacada
                    st.success(f"El valor predicho de la casa es: ${predictions[0]:,.2f}") # Formato de moneda

            else:
                 st.warning("El modelo cargado no tiene el atributo 'n_features_in_'. No se pudo verificar la consistencia de las características. Intentando predecir de todas formas...")
                 # Intentar predecir de todas formas, pero con precaución
                 try:
                     predictions = loaded_model.predict(processed_df)
                     st.subheader("Predicción del Valor de la Casa:")
                     st.success(f"El valor predicho de la casa es: ${predictions[0]:,.2f}")

                 except Exception as e:
                     st.error(f"Error al realizar predicciones: {e}")


        else:
            st.warning("No se pudo cargar el modelo o preprocesar los datos para realizar predicciones.")

    except Exception as e:
        st.error(f"Ocurrió un error durante el preprocesamiento o la predicción: {e}")

elif not models_loaded:
    st.warning("Los modelos y preprocesadores no se cargaron correctamente. No se pueden realizar predicciones.")

# Mostrar los datos de entrada generados por los widgets (opcional, para depuración)
# st.sidebar.subheader("Datos de Entrada Generados:")
# st.sidebar.dataframe(input_df)
