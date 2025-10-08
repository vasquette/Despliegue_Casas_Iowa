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

    # Widgets para variables numéricas (ejemplos)
    input_data['MSSubClass'] = st.sidebar.selectbox('MSSubClass', options=[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190], index=0, key='MSSubClass')
    input_data['LotFrontage'] = st.sidebar.number_input('LotFrontage', min_value=0.0, value=70.0, step=1.0, key='LotFrontage')
    input_data['LotArea'] = st.sidebar.number_input('LotArea', min_value=0, value=10500, step=100, key='LotArea')
    input_data['OverallQual'] = st.sidebar.slider('OverallQual', 1, 10, 5, key='OverallQual')
    input_data['OverallCond'] = st.sidebar.slider('OverallCond', 1, 9, 5, key='OverallCond')
    input_data['YearBuilt'] = st.sidebar.number_input('YearBuilt', min_value=1800, value=1980, step=1, key='YearBuilt')
    input_data['YearRemodAdd'] = st.sidebar.number_input('YearRemodAdd', min_value=1800, value=1980, step=1, key='YearRemodAdd')
    input_data['MasVnrArea'] = st.sidebar.number_input('MasVnrArea', min_value=0.0, value=0.0, step=10.0, key='MasVnrArea')
    input_data['BsmtFinSF1'] = st.sidebar.number_input('BsmtFinSF1', min_value=0, value=500, step=10, key='BsmtFinSF1')
    input_data['BsmtFinSF2'] = st.sidebar.number_input('BsmtFinSF2', min_value=0, value=0, step=10, key='BsmtFinSF2')
    input_data['BsmtUnfSF'] = st.sidebar.number_input('BsmtUnfSF', min_value=0, value=500, step=10, key='BsmtUnfSF')
    input_data['TotalBsmtSF'] = st.sidebar.number_input('TotalBsmtSF', min_value=0, value=1000, step=10, key='TotalBsmtSF')
    input_data['1stFlrSF'] = st.sidebar.number_input('1stFlrSF', min_value=0, value=1000, step=10, key='1stFlrSF')
    input_data['2ndFlrSF'] = st.sidebar.number_input('2ndFlrSF', min_value=0, value=0, step=10, key='2ndFlrSF')
    input_data['LowQualFinSF'] = st.sidebar.number_input('LowQualFinSF', min_value=0, value=0, step=10, key='LowQualFinSF')
    input_data['GrLivArea'] = st.sidebar.number_input('GrLivArea', min_value=0, value=1000, step=10, key='GrLivArea')
    input_data['BsmtFullBath'] = st.sidebar.number_input('BsmtFullBath', min_value=0, value=0, step=1, key='BsmtFullBath')
    input_data['BsmtHalfBath'] = st.sidebar.number_input('BsmtHalfBath', min_value=0, value=0, step=1, key='BsmtHalfBath')
    input_data['FullBath'] = st.sidebar.number_input('FullBath', min_value=0, value=1, step=1, key='FullBath')
    input_data['HalfBath'] = st.sidebar.number_input('HalfBath', min_value=0, value=0, step=1, key='HalfBath')
    input_data['BedroomAbvGr'] = st.sidebar.number_input('BedroomAbvGr', min_value=0, value=3, step=1, key='BedroomAbvGr')
    input_data['KitchenAbvGr'] = st.sidebar.number_input('KitchenAbvGr', min_value=0, value=1, step=1, key='KitchenAbvGr')
    input_data['TotRmsAbvGrd'] = st.sidebar.number_input('TotRmsAbvGrd', min_value=0, value=5, step=1, key='TotRmsAbvGrd')
    input_data['Fireplaces'] = st.sidebar.number_input('Fireplaces', min_value=0, value=0, step=1, key='Fireplaces')
    input_data['GarageType'] = st.sidebar.selectbox('GarageType', options=['Attchd', 'Detchd', 'BuiltIn', 'CarPort', np.nan, 'Basment', '2Types'], index=0, key='GarageType')
    input_data['GarageYrBlt'] = st.sidebar.number_input('GarageYrBlt', min_value=1800.0, value=1980.0, step=1.0, key='GarageYrBlt')
    input_data['GarageCars'] = st.sidebar.number_input('GarageCars', min_value=0, value=1, step=1, key='GarageCars')
    input_data['GarageArea'] = st.sidebar.number_input('GarageArea', min_value=0, value=300, step=10, key='GarageArea')
    input_data['WoodDeckSF'] = st.sidebar.number_input('WoodDeckSF', min_value=0, value=0, step=10, key='WoodDeckSF')
    input_data['OpenPorchSF'] = st.sidebar.number_input('OpenPorchSF', min_value=0, value=0, step=10, key='OpenPorchSF')
    input_data['EnclosedPorch'] = st.sidebar.number_input('EnclosedPorch', min_value=0, value=0, step=10, key='EnclosedPorch')
    input_data['3SsnPorch'] = st.sidebar.number_input('3SsnPorch', min_value=0, value=0, step=10, key='3SsnPorch')
    input_data['ScreenPorch'] = st.sidebar.number_input('ScreenPorch', min_value=0, value=0, step=10, key='ScreenPorch')
    input_data['PoolArea'] = st.sidebar.number_input('PoolArea', min_value=0, value=0, step=10, key='PoolArea')
    input_data['PoolQC'] = st.sidebar.selectbox('PoolQC', options=[np.nan, 'Ex', 'Gd', 'TA', 'Fa'], index=1, key='PoolQC')
    input_data['Fence'] = st.sidebar.selectbox('Fence', options=[np.nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'], index=1, key='Fence')
    input_data['MiscFeature'] = st.sidebar.selectbox('MiscFeature', options=[np.nan, 'Shed', 'Gar2', 'Othr', 'TenC'], index=1, key='MiscFeature')
    input_data['MiscVal'] = st.sidebar.number_input('MiscVal', min_value=0, value=0, step=10, key='MiscVal')
    input_data['MoSold'] = st.sidebar.slider('MoSold', 1, 12, 6, key='MoSold')
    input_data['YrSold'] = st.sidebar.slider('YrSold', 2006, 2010, 2008, key='YrSold')
    input_data['Id'] = 0 # Un ID placeholder


    # Widgets para variables categóricas (ejemplos con opciones comunes)
    input_data['MSZoning'] = st.sidebar.selectbox('MSZoning', options=['RL', 'RM', 'C (all)', 'FV', 'RH'], index=0, key='MSZoning')
    input_data['Street'] = st.sidebar.selectbox('Street', options=['Pave', 'Grvl'], index=0, key='Street')
    input_data['Alley'] = st.sidebar.selectbox('Alley', options=[np.nan, 'Grvl', 'Pave'], index=1, key='Alley')
    input_data['LotShape'] = st.sidebar.selectbox('LotShape', options=['Reg', 'IR1', 'IR2', 'IR3'], index=0, key='LotShape')
    input_data['LandContour'] = st.sidebar.selectbox('LandContour', options=['Lvl', 'Bnk', 'HLS', 'Low'], index=0, key='LandContour')
    input_data['Utilities'] = st.sidebar.selectbox('Utilities', options=['AllPub', 'NoSeWa'], index=0, key='Utilities')
    input_data['LotConfig'] = st.sidebar.selectbox('LotConfig', options=['Inside', 'Frontage', 'Corner', 'CulDSac', 'FR2', 'FR3'], index=0, key='LotConfig')
    input_data['LandSlope'] = st.sidebar.selectbox('LandSlope', options=['Gtl', 'Mod', 'Sev'], index=0, key='LandSlope')
    input_data['Neighborhood'] = st.sidebar.selectbox('Neighborhood', options=['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst', 'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW', 'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber', 'IDOTC', 'ClearCr', 'StoneBr', 'SWISU', 'Blmngtn', 'MeadowV', 'BrDale', 'Veenker', 'NPkVill', 'Blueste'], index=0, key='Neighborhood')
    input_data['Condition1'] = st.sidebar.selectbox('Condition1', options=['Norm', 'Feedr', 'Artery', 'RRAn', 'PosN', 'RRAe', 'PosA', 'RRNn', 'RRNe'], index=0, key='Condition1')
    input_data['Condition2'] = st.sidebar.selectbox('Condition2', options=['Norm', 'Artery', 'Feedr', 'RRNn', 'RRAn', 'RRAe', 'PosN', 'PosA'], index=0, key='Condition2')
    input_data['BldgType'] = st.sidebar.selectbox('BldgType', options=['1Fam', '2FmCon', 'Duplex', 'TwnhsE', 'Twnhs'], index=0, key='BldgType')
    input_data['HouseStyle'] = st.sidebar.selectbox('HouseStyle', options=['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'], index=0, key='HouseStyle')
    input_data['RoofStyle'] = st.sidebar.selectbox('RoofStyle', options=['Gable', 'Hip', 'Flat', 'Gambrel', 'Mansard', 'Shed'], index=0, key='RoofStyle')
    input_data['RoofMatl'] = st.sidebar.selectbox('RoofMatl', options=['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Roll', 'ClyTile'], index=0, key='RoofMatl')
    input_data['Exterior1st'] = st.sidebar.selectbox('Exterior1st', options=['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'CBlock', 'BrkComm', 'AsphShn', 'VinylSd', 'ImStucc', 'Stone', 'Other'], index=0, key='Exterior1st')
    input_data['Exterior2nd'] = st.sidebar.selectbox('Exterior2nd', options=['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'BrkFace', 'Wd Sdng', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'CBlock', 'BrkComm', 'AsphShn', 'VinylSd', 'ImStucc', 'Stone', 'Other'], index=0, key='Exterior2nd')
    input_data['MasVnrType'] = st.sidebar.selectbox('MasVnrType', options=['None', 'BrkFace', 'Stone', 'BrkCmn', np.nan], index=1, key='MasVnrType')
    input_data['Foundation'] = st.sidebar.selectbox('Foundation', options=['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'], index=0, key='Foundation')
    input_data['BsmtQual'] = st.sidebar.selectbox('BsmtQual', options=['Gd', 'TA', 'Ex', 'Fa', np.nan], index=1, key='BsmtQual') # Usar index=1 para 'TA' como defecto
    input_data['BsmtCond'] = st.sidebar.selectbox('BsmtCond', options=['TA', 'Gd', 'Fa', 'Po', np.nan], index=0, key='BsmtCond') # Usar index=0 para 'TA' como defecto
    input_data['BsmtExposure'] = st.sidebar.selectbox('BsmtExposure', options=['No', 'Gd', 'Mn', 'Av', np.nan], index=0, key='BsmtExposure')
    input_data['BsmtFinType1'] = st.sidebar.selectbox('BsmtFinType1', options=['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ', np.nan], index=1, key='BsmtFinType1') # Usar index=1 para 'ALQ' como defecto
    input_data['BsmtFinType2'] = st.sidebar.selectbox('BsmtFinType2', options=['Unf', 'Rec', 'LwQ', 'GLQ', 'ALQ', 'BLQ', np.nan], index=0, key='BsmtFinType2') # Usar index=0 para 'Unf' como defecto
    input_data['Heating'] = st.sidebar.selectbox('Heating', options=['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'], index=0, key='Heating')
    input_data['HeatingQC'] = st.sidebar.selectbox('HeatingQC', options=['Ex', 'Gd', 'TA', 'Fa', 'Po'], index=0, key='HeatingQC') # Usar index=0 para 'Ex' como defecto
    input_data['CentralAir'] = st.sidebar.selectbox('CentralAir', options=['Y', 'N'], index=0, key='CentralAir')
    input_data['Electrical'] = st.sidebar.selectbox('Electrical', options=['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix', np.nan], index=1, key='Electrical')
    input_data['KitchenQual'] = st.sidebar.selectbox('KitchenQual', options=['Gd', 'TA', 'Ex', 'Fa'], index=1, key='KitchenQual') # Usar index=1 para 'TA' como defecto
    input_data['Functional'] = st.sidebar.selectbox('Functional', options=['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev'], index=0, key='Functional') # Usar index=0 para 'Typ' como defecto
    input_data['FireplaceQu'] = st.sidebar.selectbox('FireplaceQu', options=[np.nan, 'Gd', 'TA', 'Fa', 'Ex', 'Po'], index=1, key='FireplaceQu')
    input_data['PoolArea'] = st.sidebar.number_input('PoolArea', min_value=0, value=0, step=10, key='PoolArea')
    input_data['PoolQC'] = st.sidebar.selectbox('PoolQC', options=[np.nan, 'Ex', 'Gd', 'TA', 'Fa'], index=1, key='PoolQC')
    input_data['Fence'] = st.sidebar.selectbox('Fence', options=[np.nan, 'MnPrv', 'GdWo', 'GdPrv', 'MnWw'], index=1, key='Fence')
    input_data['MiscFeature'] = st.sidebar.selectbox('MiscFeature', options=[np.nan, 'Shed', 'Gar2', 'Othr', 'TenC'], index=1, key='MiscFeature')
    input_data['MiscVal'] = st.sidebar.number_input('MiscVal', min_value=0, value=0, step=10, key='MiscVal')
    input_data['MoSold'] = st.sidebar.slider('MoSold', 1, 12, 6, key='MoSold')
    input_data['YrSold'] = st.sidebar.slider('YrSold', 2006, 2010, 2008, key='YrSold')
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
                 df_processed[col] = df_processed[col].fillna('Missing') # Fill NaN with 'Missing' string

            # Convertir a tipo string para asegurar consistencia antes de OHE
            df_processed[ohe_cols_in_df] = df_processed[ohe_cols_in_df].astype(str)

            # Add 'Missing' to the categories of the one-hot encoder if it's not already there.
            # This requires checking if the encoder was trained with handle_unknown='ignore'.
            # If not, we need to manually add the category or use a different strategy.
            # Assuming the loaded encoder might not have handle_unknown='ignore' and we need to handle 'Missing'.
            # A robust way is to re-fit a new encoder on the training data categories PLUS 'Missing',
            # but that requires access to training data categories which we don't have here.
            # A simpler approach is to ensure the encoder supports unknown values.
            # If the loaded encoder doesn't support handle_unknown='ignore', the next line will fail.
            # Let's assume it *does* or we re-trained it to.

            # Use handle_unknown='ignore' if the encoder was trained with this option.
            # If the loaded encoder doesn't support handle_unknown='ignore', we would need
            # a more complex strategy (e.g., re-fitting the encoder with the union of categories)
            # For now, assume handle_unknown='ignore' is supported or the encoder was trained to handle 'Missing'.

            # Let's explicitly add 'Missing' to categories if it's not present in the encoder's categories
            # This is a more direct way to handle the ValueError.
            for col in ohe_cols_in_df:
                 if 'Missing' not in onehot_encoder.categories_[categorical_cols_ohe.index(col)]:
                     # This is tricky with a loaded encoder. We cannot simply add categories to a fitted encoder.
                     # A better approach is to ensure the input data is prepared correctly *before* transforming.
                     # The fillna('Missing').astype(str) part is already done.
                     # The ValueError "Cannot setitem on a Categorical with a new category" suggests
                     # that the column itself might still be a Pandas Categorical type with fixed categories,
                     # and 'Missing' is not in those categories.
                     # Let's ensure it's a plain string dtype before OHE. The .astype(str) above should help.
                     pass # The .astype(str) should handle this, the ValueError might be from the encoder itself.

            # Applying One-Hot Encoding with handle_unknown='ignore' if possible
            try:
                encoded_data = onehot_encoder.transform(df_processed[ohe_cols_in_df])
            except ValueError as e:
                 # If handle_unknown='ignore' wasn't used during training, this might still fail.
                 # Fallback: Re-initialize a compatible encoder with handle_unknown='ignore'
                 st.warning(f"One-Hot Encoder might not support unknown categories. Trying to re-initialize with handle_unknown='ignore'. Original error: {e}")
                 try:
                     from sklearn.preprocessing import OneHotEncoder
                     # Get the categories from the loaded encoder
                     known_categories = onehot_encoder.categories_
                     # Re-create an encoder with the known categories and handle_unknown='ignore'
                     temp_ohe = OneHotEncoder(categories=known_categories, handle_unknown='ignore', sparse_output=onehot_encoder.sparse_output)
                     encoded_data = temp_ohe.transform(df_processed[ohe_cols_in_df])
                     st.success("Successfully re-initialized and applied One-Hot Encoder with handle_unknown='ignore'.")
                 except Exception as re_init_e:
                     st.error(f"Failed to re-initialize One-Hot Encoder. One-Hot Encoding cannot be applied. Error: {re_init_e}")
                     encoded_data = None # Indicate failure

            if encoded_data is not None:
                # Convert the result to array denso if is a sparse matrix
                if isinstance(encoded_data, scipy.sparse.csr.csr_matrix):
                     encoded_data = encoded_data.toarray()


                # Crear un DataFrame con las columnas codificadas
                # Usar get_feature_names_out con las columnas del DataFrame de entrada para evitar errores de coincidencia
                # If handle_unknown='ignore' was used, get_feature_names_out might not include columns for ignored categories.
                # Use the feature names from the original encoder if possible.
                try:
                     encoded_column_names = onehot_encoder.get_feature_names_out(ohe_cols_in_df)
                except AttributeError:
                     # Fallback if get_feature_names_out is not available (older sklearn versions)
                     st.warning("Using feature names from the re-initialized encoder (may not be perfectly aligned).")
                     encoded_column_names = temp_ohe.get_feature_names_out(ohe_cols_in_df) # Use names from temp encoder


                encoded_df = pd.DataFrame(encoded_data, columns=encoded_column_names, index=df_processed.index)

                # Eliminar las columnas categóricas originales
                df_processed = df_processed.drop(columns=ohe_cols_in_df)

                # Concatenar el DataFrame original con las columnas codificadas
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


    # 4. Escalar columnas numéricas (asegúrate de que las columnas existan y manejar valores NaN después de OHE if applies)
    numerical_cols_to_scale = ['OverallQual', 'OverallCond','LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF']
    scale_cols_in_df = [col for col in numerical_cols_to_scale if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]

    if scale_cols_in_df:
        # Rellenar posibles NaN con la media o un valor por defecto antes de escalar si es necesario
        # Aquí usamos 0, but it's better to use the mean from the training set
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
