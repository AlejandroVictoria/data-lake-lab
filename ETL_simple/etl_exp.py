# file_uploader.py MinIO Python SDK example
from minio import Minio
from minio.error import S3Error
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import trino


def uploading_data(source_file, destination_file, bucket_name, client):
    # Make the bucket if it doesn't exist.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # Upload the file, renaming it in the process
    client.fput_object(
        bucket_name, destination_file, source_file,
    )
    print(
        source_file, "successfully uploaded as object",
        destination_file, "to bucket", bucket_name,
    )

    
def schematize_data(bucket_name, folder_name, schema_name, table_name, client):
    cursor = client.cursor()
    print("Creating schema...")
    cursor.execute(f"""CREATE SCHEMA IF NOT EXISTS minio.{schema_name}
        with (location = 's3a://{bucket_name}/{folder_name}/')""")

    # 1. Eliminar tabla previa para limpiar el metadata de Trino
    cursor.execute(f"DROP TABLE IF EXISTS minio.{schema_name}.{table_name}")

    # 2. Crear la tabla apuntando a la ubicación del experimento actual
    query = f"""
    CREATE TABLE minio.{schema_name}.{table_name} (
        VendorID BIGINT, -- Cambiado de INTEGER
        tpep_pickup_datetime TIMESTAMP(6),
        tpep_dropoff_datetime TIMESTAMP(6),
        passenger_count BIGINT, -- Los pasajeros suelen ser enteros en Parquet
        trip_distance DOUBLE,
        RatecodeID BIGINT, -- Cambiado de DOUBLE
        store_and_fwd_flag VARCHAR,
        PULocationID BIGINT, -- Cambiado de INTEGER
        DOLocationID BIGINT, -- Cambiado de INTEGER
        payment_type BIGINT,
        fare_amount DOUBLE,
        extra DOUBLE,
        mta_tax DOUBLE,
        tip_amount DOUBLE,
        tolls_amount DOUBLE,
        improvement_surcharge DOUBLE,
        total_amount DOUBLE,
        congestion_surcharge DOUBLE,
        Airport_fee DOUBLE,
        cbd_congestion_fee DOUBLE
    )
    WITH (
        external_location = 's3a://{bucket_name}/{folder_name}/',
        format = 'PARQUET'
    )
    """

    cursor.execute(query)
    cursor.close()

if __name__ == "__main__":
    try:
        minio_client = Minio(
            endpoint= "localhost:9000",
            access_key="minio_access_key",
            secret_key="minio_secret_key",
            secure=False
        )
        trino_client = trino.dbapi.connect(
            host="localhost",
            port=8086,
            user="trino",
            catalog="minio",
            schema="default"
        )

        print("uploading data...")
        source_file = "data/raw_data/yellow_tripdata_2026-01.parquet"
        destination_file = "nyc_taxi.parquet"
        bucket_name = "landing-zone"
        uploading_data(source_file, destination_file, bucket_name, minio_client)

        print("schematizing data...")
        schema_name = "nyc_taxi_schema"
        table_name = "nyc_taxi_trips"
        schematize_data(bucket_name, schema_name, table_name, trino_client)

    except S3Error as exc:
        print("error occurred.", exc)
