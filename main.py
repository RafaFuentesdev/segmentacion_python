from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler, PCA


# Initialize Spark session
spark = SparkSession.builder \
    .appName("PCA Dimensionality Reduction") \
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

# Apply PCA for dimensionality reduction
num_pca_components = 3 # Change this value according to your needs
pca = PCA(k=num_pca_components, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(scaled_data)
pca_data = pca_model.transform(scaled_data)

# Show the first 10 rows with the PCA features
pca_data.select("pca_features").show(10)

# Perform any further data processing as needed
# ...
explained_variance = pca_model.explainedVariance
print("Explained Variance by Principal Components:")
sum = 0
for i, variance in enumerate(explained_variance):
    print(f"PC{i+1}: {variance:.4f}")
    sum += variance
print(f"Total variance {(sum*100):.4f}%")

# Stop Spark session
spark.stop()
