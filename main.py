from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Data Preprocessing") \
    .getOrCreate()

# Read data from Parquet file
data = spark.read.parquet("data.parquet")

# Define the columns to be standardized (excluding the 'id' and 'estaciones' columns)
columns_to_standardize = [
    "capacidad_campo_media", "pendiente_3clases", "porosidad_media",
    "punto_marchitez_medio", "umbral_humedo", "umbral_intermedio", "umbral_seco"
]

# Assemble the columns into a single vector column
assembler = VectorAssembler(inputCols=columns_to_standardize, outputCol="features")
data_with_features = assembler.transform(data)

# Standardize the features using StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(data_with_features)
scaled_data = scaler_model.transform(data_with_features)

# Show the first 10 rows with the original and standardized features
scaled_data.select("features", "scaled_features").show(10)


# Perform any further data processing as needed
# ...

# Stop Spark session
spark.stop()
