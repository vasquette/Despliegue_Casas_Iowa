import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Cargar Modelos y Preprocesadores ---
# Intenta cargar cada archivo y maneja los errores si no se encuentran
try:
    loaded_scaler = joblib.load('minmax_scaler.pkl')
    overallqual_label_encoder = joblib.load('OverallQual_label_encoder.pkl')
    overallcond_label_encoder = joblib.load('OverallCond_label_encoder.pkl')
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


# --- Función para Generar Datos por Defecto (80 variables) ---
def get_default_input_data():
    """Genera un DataFrame con valores por defecto para las 80 variables."""
    # Aquí definimos valores comunes o representativos para un conjunto típico de variables
    # Puedes ajustar estos valores según la distribución de tus datos de entrenamiento
    data = {
        'Id': [0], # Placeholder ID
        'MSSubClass': [20],
        'MSZoning': ['RL'],
        'LotFrontage': [70.0], # Usar un valor común o la media
        'LotArea': [10500], # Usar un valor común o la media
        'Street': ['Pave'],
        'Alley': [np.nan], # Usar NaN para valores faltantes comunes
        'LotShape': ['Reg'],
        'LandContour': ['Lvl'],
        'Utilities': ['AllPub'],
        'LotConfig': ['Inside'],
        'LandSlope': ['Gtl'],
        'Neighborhood': ['NAmes'],
        'Condition1': ['Norm'],
        'Condition2': ['Norm'],
        'BldgType': ['1Fam'],
        'HouseStyle': ['1Story'],
        'OverallQual': [5], # Usar un valor numérico representativo
        'OverallCond': [5], # Usar un valor numérico representativo
        'YearBuilt': [1980],
        'YearRemodAdd': [1980],
        'RoofStyle': ['Gable'],
        'RoofMatl': ['CompShg'],
        'Exterior1st': ['VinylSd'],
        'Exterior2nd': ['VinylSd'],
        'MasVnrType': ['None'],
        'MasVnrArea': [0.0],
        'ExterQual': ['TA'], # Usar la representación de cadena esperada por el encoder
        'ExterCond': ['TA'], # Usar la representación de cadena esperada por el encoder
        'Foundation': ['CBlock'],
        'BsmtQual': ['TA'], # Usar la representación de cadena esperada por el encoder
        'BsmtCond': ['TA'], # Usar la representación de cadena esperada por el encoder
        'BsmtExposure': ['No'], # Usar la representación de cadena esperada por el encoder
        'BsmtFinType1': ['ALQ'], # Usar la representación de cadena esperada por el encoder
        'BsmtFinSF1': [500],
        'BsmtFinType2': ['Unf'], # Usar la representación de cadena esperada por el encoder
        'BsmtFinSF2': [0],
        'BsmtUnfSF': [500],
        'TotalBsmtSF': [1000],
        'Heating': ['GasA'],
        'HeatingQC': ['Ex'], # Usar la representación de cadena esperada por el encoder
        'CentralAir': ['Y'], # Usar la representación de cadena esperada por el encoder
        'Electrical': ['SBrkr'], # Usar la representación de cadena esperada por el encoder
        '1stFlrSF': [1000],
        '2ndFlrSF': [0],
        'LowQualFinSF': [0],
        'GrLivArea': [1000],
        'BsmtFullBath': [0],
        'BsmtHalfBath': [0],
        'FullBath': [1],
        'HalfBath': [0],
        'BedroomAbvGr': [3],
        'KitchenAbvGr': [1],
        'KitchenQual': ['TA'], # Usar la representación de cadena esperada por el encoder
        'TotRmsAbvGrd': [5],
        'Functional': ['Typ'], # Usar la representación de cadena esperada por el encoder
        'Fireplaces': [0],
        'FireplaceQu': [np.nan], # Usar NaN para valores faltantes comunes
        'GarageType': ['Attchd'], # Usar la representación de cadena esperada por el encoder
        'GarageYrBlt': [1980.0],
        'GarageFinish': ['Unf'], # Usar la representación de cadena esperada por el encoder
        'GarageCars': [1],
        'GarageArea': [300],
        'GarageQual': [np.nan], # Usar NaN para valores faltantes comunes
        'GarageCond': [np.nan], # Usar NaN para valores faltantes comunes
        'PavedDrive': ['Y'], # Usar la representación de cadena esperada por el encoder
        'WoodDeckSF': [0],
        'OpenPorchSF': [0],
        'EnclosedPorch': [0],
        '3SsnPorch': [0],
        'ScreenPorch': [0],
        'PoolArea': [0],
        'PoolQC': [np.nan], # Usar NaN para valores faltantes comunes
        'Fence': [np.nan], # Usar NaN para valores faltantes comunes
        'MiscFeature': [np.nan], # Usar NaN para valores faltantes comunes
        'MiscVal': [0],
        'MoSold': [6],
        'YrSold': [2008],
        'SaleType': ['WD'], # Usar la representación de cadena esperada por el encoder
        'SaleCondition': ['Normal'] # Usar la representación de cadena esperada por el encoder
    }
    return pd.DataFrame(data)


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
            st.warning(f"Columna '{col}' faltante en los datos de entrada. Añadida con NaN.")


    # Seleccionar solo las columnas que el modelo espera (las que se usaron en el entrenamiento)
    df_processed = df_processed[expected_original_cols].copy()


    # 1. Convertir a categoría si es necesario (asegúrate de que las columnas existan)
    # Identificar columnas categóricas en el DataFrame actual
    current_categorical_cols = df_processed.select_dtypes(include='object').columns
    for col in current_categorical_cols:
         if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')


    # 2. Aplicar LabelEncoder (asegúrate de manejar etiquetas no vistas)
    label_encode_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'Functional', 'GarageFinish', 'OverallQual', 'OverallCond'] # Agregadas OverallQual y OverallCond
    label_encoders = {
        'OverallQual': overallqual_label_encoder,
        'OverallCond': overallcond_label_encoder,
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
                 # Asegurarse de que todas las categorías del encoder estén presentes en la columna antes de transformar
                 # y manejar etiquetas no vistas si las hay
                 current_categories = set(df_processed[col].cat.categories)
                 encoder_classes = set(encoder.classes_)
                 unseen_in_input = list(current_categories - encoder_classes)

                 if unseen_in_input:
                     st.warning(f"Etiquetas no vistas en la columna '{col}': {unseen_in_input}. Reemplazando con NaN.")
                     # Reemplazar etiquetas no vistas con NaN para que sean tratadas como valores faltantes
                     df_processed[col] = df_processed[col].replace(unseen_in_input, np.nan)


                 # Convertir a tipo category con las categorías conocidas por el encoder
                 # Esto asegura que la transformación funcione incluso si no todas las categorías están en el input actual
                 df_processed[col] = df_processed[col].astype(pd.CategoricalDtype(categories=encoder.classes_))

                 # Aplicar la transformación (esto convertirá las categorías a sus códigos numéricos)
                 df_processed[col] = df_processed[col].cat.codes
                 # Los valores que eran NaN o etiquetas no vistas (reemplazados por NaN) se convertirán a -1 por .cat.codes
                 # Podemos rellenar estos -1 con un valor por defecto si es necesario, ej. la moda del entrenamiento
                 # Por ahora, los dejaremos como -1 o NaN si no fueron convertidos a category type


             except Exception as e:
                 st.warning(f"No se pudo aplicar LabelEncoder a la columna '{col}': {e}")
                 # Si falla, intentar convertir a numérico y rellenar NaN
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(-1) # Rellenar con -1 si la conversión falla o hay NaN


    # Convertir columnas de LabelEncoder a tipo numérico si no lo son ya
    for col in label_encode_cols:
        if col in df_processed.columns:
             if not pd.api.types.is_numeric_dtype(df_processed[col]):
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                 df_processed[col] = df_processed[col].fillna(-1)


    # 3. Aplicar One-Hot Encoding (asegúrate de que las columnas existan y manejar columnas faltantes/extra)
    categorical_cols_ohe = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'Exterior1st', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition']
    ohe_cols_in_df = [col for col in categorical_cols_ohe if col in df_processed.columns]

    if ohe_cols_in_df:
        try:
            # Aplicar One-Hot Encoding
            # Usar handle_unknown='ignore' si el encoder fue entrenado con esta opción,
            # o asegurarse de que las categorías del input coincidan con las del encoder.
            # Para este ejemplo, asumiremos que el encoder maneja columnas y categorías conocidas.
            # Si hay categorías no vistas, el transform() por defecto lanzará un error a menos que se manejen.
            # Una forma es usar una versión del encoder que soporte handle_unknown='ignore' o
            # asegurarse de que el input solo tenga categorías vistas.
            # Dada la carga previa del encoder, asumimos que es un OneHotEncoder de sklearn que soporta handle_unknown='ignore' en versiones recientes.
            # Si no, necesitarías actualizar scikit-learn o implementar manejo manual.
            # Aquí, intentaremos transformar directamente. Si hay error, se capturará.
            encoded_data = onehot_encoder.transform(df_processed[ohe_cols_in_df])

            # Crear un DataFrame con las columnas codificadas
            encoded_df = pd.DataFrame(encoded_data.toarray(), columns=onehot_encoder.get_feature_names_out(ohe_cols_in_df), index=df_processed.index)

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
             st.error(f"El número de características después de la alineación ({df_aligned.shape[1]}) no coincide con las esperadas por el modelo PCA ({pca_model.n_features_in_}). No se aplicará PCA.")
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

st.write("Esta aplicación predice el valor de una casa basado en sus características. Puedes subir un archivo CSV o usar los datos por defecto para una predicción rápida.")

# Opción para usar datos por defecto o cargar archivo CSV
prediction_method = st.radio("Selecciona el método de entrada de datos:", ('Usar datos por defecto', 'Subir archivo CSV'))

input_df = None # Inicializar input_df

if prediction_method == 'Usar datos por defecto':
    st.write("Usando datos por defecto para la predicción.")
    input_df = get_default_input_data()
    st.write("Datos de entrada por defecto:")
    st.dataframe(input_df) # Mostrar los datos por defecto


elif prediction_method == 'Subir archivo CSV':
    uploaded_file = st.file_uploader("Sube un archivo CSV con los datos de las casas a predecir", type=["csv"])
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("Datos cargados:")
            st.dataframe(input_df.head())
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo CSV: {e}")


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
                else:
                    predictions = loaded_model.predict(processed_df)
                    st.write("Predicciones de valor de casas:")
                    # Mostrar las predicciones, quizás junto con un identificador si el archivo de entrada tenía uno (ej: 'Id')
                    if 'Id' in input_df.columns:
                        prediction_output = input_df[['Id']].copy()
                        prediction_output['Predicted_SalePrice'] = predictions
                        st.dataframe(prediction_output)
                    else:
                        st.write(predictions)

            else:
                 st.warning("El modelo cargado no tiene el atributo 'n_features_in_'. No se pudo verificar la consistencia de las características. Intentando predecir de todas formas...")
                 # Intentar predecir de todas formas, pero con precaución
                 try:
                     predictions = loaded_model.predict(processed_df)
                     st.write("Predicciones de valor de casas:")
                     if 'Id' in input_df.columns:
                        prediction_output = input_df[['Id']].copy()
                        prediction_output['Predicted_SalePrice'] = predictions
                        st.dataframe(prediction_output)
                     else:
                        st.write(predictions)
                 except Exception as e:
                     st.error(f"Error al realizar predicciones: {e}")


        else:
            st.warning("No se pudo cargar el modelo o preprocesar los datos para realizar predicciones.")

    except Exception as e:
        st.error(f"Ocurrió un error durante el preprocesamiento o la predicción: {e}")

elif not models_loaded:
    st.warning("Los modelos y preprocesadores no se cargaron correctamente. No se pueden realizar predicciones.")
