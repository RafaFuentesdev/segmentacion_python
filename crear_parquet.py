from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CSV Data Processing") \
    .getOrCreate()

# Define schema for the CSV data
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("capacidad_campo_media", FloatType(), True),
    StructField("estaciones", IntegerType(), True),
    StructField("pendiente_3clases", FloatType(), True),
    StructField("porosidad_media", FloatType(), True),
    StructField("punto_marchitez_medio", FloatType(), True),
    StructField("umbral_humedo", FloatType(), True),
    StructField("umbral_intermedio", FloatType(), True),
    StructField("umbral_seco", FloatType(), True)
])

# Read CSV data using the defined schema
data = spark.read.csv("completo0.csv", header=True, schema=schema)
# Write data to Parquet format
data.write.parquet("data.parquet")

# Show the first 10 rows
data.show(10)

# Perform any further data processing as needed
# ...

# Stop Spark session
spark.stop()
