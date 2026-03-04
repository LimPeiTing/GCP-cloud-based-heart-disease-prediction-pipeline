import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# --- Configuration (Must be updated for your environment) ---
GCS_FILE_PATH = "gs://cardiopredict-bronzedata-12345/heart_2022_no_nans.csv"
BIGQUERY_OUTPUT_TABLE = "cardio-predict-project.heart_analytics_dataset.processed_heart_data"
GCS_STAGING_BUCKET = "cardiopredict-bronzedata-12345" # Used for BigQuery temporary files

if len(sys.argv) > 1:
    GCS_FILE_PATH = sys.argv[1]
    print(f"Using GCS Path from command line argument: {GCS_FILE_PATH}")


def main():
    """
    Initializes Spark, performs ETL, and writes the final de-normalized DataFrame to BigQuery.
    """
    # 1. Initialize PySpark Session
    spark = SparkSession.builder.appName("HeartDataPrepBQ").getOrCreate()

    # Set the GCS staging bucket configuration for the BigQuery connector
    spark.conf.set("temporaryGcsBucket", GCS_STAGING_BUCKET)
    print("Spark Session Initialized on Dataproc.")
    print(f"BigQuery output table set to: {BIGQUERY_OUTPUT_TABLE}")

    # --- 2. Data Loading, Cleaning, and Preprocessing ---
    try:
        # Load data from GCS
        df = spark.read.csv(GCS_FILE_PATH, header=True, inferSchema=True)
        print(f"Data loaded successfully. Records: {df.count()}")

        # Ensure numerical types for calculations
        df = df.withColumn("WeightInKilograms", F.col("WeightInKilograms").cast("double"))
        df = df.withColumn("HeightInMeters", F.col("HeightInMeters").cast("double"))

        # Feature Engineering: Create BMI (Weight / Height^2)
        df = df.withColumn(
            "BMI_Calculated",
            F.col("WeightInKilograms") / (F.col("HeightInMeters") ** 2)
        )
        # Drop calculated invalid/missing BMI rows
        df = df.filter(F.col("BMI_Calculated").isNotNull() & ~F.isnan(F.col("BMI_Calculated")))

        # Target Pre-processing: Convert 'Yes'/'No' target into numerical 1/0
        df = df.withColumn("HadHeartAttack_Numeric", F.when(F.col("HadHeartAttack") == "Yes", 1).otherwise(0))

        # Define feature columns and drops
        categorical_cols = [
            'Sex', 'GeneralHealth', 'LastCheckupTime', 'PhysicalActivities', 'RemovedTeeth',
            'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
            'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
            'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating',
            'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands',
            'SmokerStatus', 'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory',
            'AgeCategory', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12',
            'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos'
        ]
        numerical_cols = [
            'PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'BMI_Calculated'
        ]
        drop_cols = ['State', 'HeightInMeters', 'WeightInKilograms', 'BMI', 'HadHeartAttack']
        df = df.drop(*drop_cols)

    except Exception as e:
        print(f"ERROR during data loading or cleaning: {e}")
        spark.stop()
        return

    # --- 3. ML Pipeline for Transformations ---

    # Stage 1: Indexing Categorical Variables (Converts strings to numerical indices)
    indexers = [
        StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid="skip")
        for col in categorical_cols
    ]
    indexed_cols = [col + "_indexed" for col in categorical_cols]

    # Stage 2: Scaling Numerical Features (Uses StandardScaler on the raw numerical columns for standardization)
    # We assemble only the numerical columns first to scale them
    num_assembler = VectorAssembler(inputCols=numerical_cols, outputCol="num_features_unscaled", handleInvalid="skip")
    num_scaler = StandardScaler(inputCol="num_features_unscaled", outputCol="num_features_scaled", withStd=True, withMean=False)

    # --- Combine transformations into a pipeline ---
    # The pipeline will apply indexing, then scale the numerical data.
    transform_pipeline = Pipeline(stages=indexers + [num_assembler, num_scaler])

    # Apply the transformations
    model = transform_pipeline.fit(df)
    processed_df = model.transform(df)

    # --- 4. Write Cleaned & Engineered Data to BigQuery (De-vectorized) ---
    print("\n--- 4. Writing Processed Data to BigQuery ---")

    # Define the final columns for BigQuery: Label + Individual Features
    bq_cols = ["HadHeartAttack_Numeric"] + indexed_cols

    # NOTE: The scaled numerical features are currently in a vector column ('num_features_scaled').
    # To write a flat table, we will select the original numerical columns,
    # and BigQuery ML can handle normalization/scaling later. If scaling is essential
    # to be in the BQ table, we would need to manually de-vectorize the 'num_features_scaled' column,
    # but for simplicity and BigQuery ML readiness, we will write the raw numerical values.

    # Final BQ columns: Label + Indexed Categorical + Raw Numerical
    final_bq_cols = ["HadHeartAttack_Numeric"] + indexed_cols + numerical_cols

    # Select these simple columns for a flat, clean BigQuery table
    final_df_bq = processed_df.select(*final_bq_cols)

    print(f"Writing {final_df_bq.count()} records to BigQuery...")
    final_df_bq.printSchema()

    try:
        # Write to BigQuery using the Spark connector
        final_df_bq.write \
            .format("bigquery") \
            .option("table", BIGQUERY_OUTPUT_TABLE) \
            .mode("overwrite") \
            .save()

        print("\nSuccessfully wrote processed data to BigQuery!")
        print(f"Table Name: {BIGQUERY_OUTPUT_TABLE}")

    except Exception as e:
        print(f"ERROR: Could not write to BigQuery. Please check permissions, table path, and connector setup.")
        print(e)

    spark.stop()


if __name__ == "__main__":
    main()
