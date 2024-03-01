import numpy as np
import pandas as pd
import requests

def descargar_procesar_guardar_csv(url, nombre_archivo_salida):
    try:
        # Descarga del archivo CSV
        respuesta = requests.get(url)
        respuesta.raise_for_status()   
        
        nombre_archivo_temporal = "temporal.csv"
        with open(nombre_archivo_temporal, 'wb') as archivo:
            archivo.write(respuesta.content)
        print(f"Archivo descargado exitosamente y guardado en {nombre_archivo_temporal}")
        
        # Carga del archivo CSV en un DataFrame
        dataframe = pd.read_csv(nombre_archivo_temporal)
        
        # Procesamiento del DataFrame
        # Eliminar valores faltantes y filas duplicadas
        dataframe.dropna(inplace=True)
        dataframe.drop_duplicates(inplace=True)
        
        # Eliminar valores atípicos en columnas numéricas relevantes
        cols_numericas = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
        Q1 = dataframe[cols_numericas].quantile(0.25)
        Q3 = dataframe[cols_numericas].quantile(0.75)
        IQR = Q3 - Q1
        dataframe = dataframe[~((dataframe[cols_numericas] < (Q1 - 1.5 * IQR)) | (dataframe[cols_numericas] > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        # Categorizar por edades
        bins = [0, 12, 19, 39, 59, np.inf]
        labels = ['Niño', 'Adolescente', 'Jóvenes adulto', 'Adulto', 'Adulto mayor']
        dataframe['Categoria_edad'] = pd.cut(dataframe['age'], bins=bins, labels=labels, right=False)
        nombre_archivo_salida = "heart_failure_dataset_procesado.csv"
        
        # Guardar el DataFrame procesado
        dataframe.to_csv(nombre_archivo_salida, index=False)
        print(f"DataFrame procesado y guardado en {nombre_archivo_salida}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar los datos: {e}")

 
url = 'https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv'
nombre_archivo_salida = "/mnt/data/heart_failure_dataset_procesado.csv"

 
descargar_procesar_guardar_csv(url, nombre_archivo_salida)
