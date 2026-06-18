# This script is for creating a pipeline for consumption data from MinIO, processing it,
# and then loading it into Trino for querying and then visualize it or train a model with it.
import pandas as pd
import datetime as dt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import trino


import matplotlib.pyplot as plt
import seaborn as sns


def featurize_data(df):
    """
    Agrega la demanda por zona/hora y crea el TimeIndex.
    """
    # 1. Limpieza rápida de calidad (Esencial en el dataset de Taxi)
    df = df[(df['trip_distance'] > 0) & (df['total_amount'] > 0)]
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    
    # 2. Agregación de demanda global por zona y hora
    df_demand = df.groupby([
        'pulocationid', 
        pd.Grouper(key='tpep_pickup_datetime', freq='h')
    ]).size().reset_index(name='trip_count')
    
    # 3. Creación del TimeIndex
    anchor_date = df_demand['tpep_pickup_datetime'].min()
    df_demand['TimeIndex'] = ((df_demand['tpep_pickup_datetime'] - anchor_date) / pd.Timedelta(hours=1)).astype(int)
    
    # 4. Features adicionales (Cíclicas) para el modelo global
    df_demand['hour'] = df_demand['tpep_pickup_datetime'].dt.hour
    df_demand['day_of_week'] = df_demand['tpep_pickup_datetime'].dt.dayofweek
    
    return df_demand, anchor_date


def train_and_evaluate_pipeline(df_f):
    """
    Entrena un modelo global para predecir la demanda de taxis.
    """
    # Definir Features (X) y Target (y)
    # Incluimos la zona y el tiempo para que el modelo sea 'Global'
    features = ['pulocationid', 'TimeIndex', 'hour', 'day_of_week']
    X = df_f[features]
    y = df_f['trip_count']
    
    # Split cronológico (80% entrenamiento, 20% prueba)
    split_point = int(len(df_f) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # Modelo: RandomForest es excelente para capturar patrones no lineales en zonas
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluación
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"Evaluación Global - RMSE: {rmse:.2f}, R2: {r2:.2f}")
    
    return {"model": model, "rmse": rmse, "r2": r2, "features": features}


def visualize_taxi_demand(df_f, model_results, zone_id=161):
    """
    Visualiza la demanda real vs la predicción para una zona específica (ej. Midtown Manhattan).
    """
    
    # 1. Filtrar datos para la zona elegida
    df_zone = df_f[df_f['pulocationid'] == zone_id].copy()
    
    # 2. Generar predicciones solo para esa zona
    features = model_results['features']
    df_zone['prediction'] = model_results['model'].predict(df_zone[features])
    
    # 3. Graficar
    plt.figure(figsize=(12, 6))
    plt.plot(df_zone['tpep_pickup_datetime'], df_zone['trip_count'], label='Demanda Real', alpha=0.7)
    plt.plot(df_zone['tpep_pickup_datetime'], df_zone['prediction'], label='Predicción Regresión', linestyle='--')
    
    plt.title(f'Análisis de Serie de Tiempo - Demanda Zona {zone_id}')
    plt.xlabel('Fecha y Hora')
    plt.ylabel('Número de Viajes')
    plt.legend()
    plt.show()


def consumption_pipeline(trino_conn, schema_name, table_name):
    """
    Flujo completo: Extracción -> Featurización -> Entrenamiento.
    """
    query = f"""
        SELECT 
            pulocationid, 
            tpep_pickup_datetime, 
            trip_distance, 
            total_amount 
        FROM minio.{schema_name}.{table_name}
        """
    
    # Medir el tiempo de lectura (Parte crítica del Benchmark)
    print("Reading data from Trino...")
    df = pd.read_sql_query(query, trino_conn)
    
    # Ingeniería de variables
    print("Featurizing data...")
    df_featured, anchor = featurize_data(df)
    
    # Entrenamiento del modelo global
    print("Training model...")
    results = train_and_evaluate_pipeline(df_featured)

    # Visualización para una zona específica
    visualize_taxi_demand(df_featured, results, zone_id=161)
    
    return results, anchor

if __name__=="__main__":
    # Access config for MinIO
    trino_client = trino.dbapi.connect(
        host="localhost",
        port=8086,
        user="trino",
        catalog="minio",
        schema="default"
    )
    schema_name = "nyc_taxi_schema"
    table_name = "nyc_taxi_trips"

    data = consumption_pipeline(trino_client, schema_name, table_name)