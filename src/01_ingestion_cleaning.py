# ============================================================
# AMAZON REVIEW BIG DATA PROJECT
# PERSON 1: DATA INGESTION & CLEANING
# ============================================================

# ============================================================
# 1. PYTHON ENVIRONMENT CONFIGURATION
# ============================================================

import os
import sys

# Make Spark use the Python from the current virtual environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


# ============================================================
# 2. IMPORT LIBRARIES
# ============================================================

from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    sum,
    when,
    from_unixtime,
    year,
    month,
    length,
    trim
)

import json


# ============================================================
# 3. CREATE SPARK SESSION
# ============================================================

spark = SparkSession.builder \
    .appName("AmazonReviewDataCleaning") \
    .master("local[*]") \
    .getOrCreate()


# ============================================================
# 4. LOAD RAW AMAZON REVIEW DATA
# ============================================================

print("\n========================================")
print("LOADING AMAZON REVIEW DATA")
print("========================================")

reviews = spark.read.json(
    "data/Musical_Instruments.jsonl"
)


# ============================================================
# 5. ORIGINAL DATA INFORMATION
# ============================================================

print("\n========== ORIGINAL DATA ==========")

original_count = reviews.count()

print("Number of reviews:", original_count)

print("Number of columns:", len(reviews.columns))

print("\nColumns:")
print(reviews.columns)

print("\nSchema:")
reviews.printSchema()


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing_values = reviews.select([
    sum(
        when(
            col(c).isNull(),
            1
        ).otherwise(0)
    ).alias(c)
    for c in reviews.columns
])

missing_values.show()


# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

print("\n========== REMOVING DUPLICATES ==========")

before_duplicates = reviews.count()

reviews = reviews.dropDuplicates()

after_duplicates = reviews.count()

print(
    "Reviews before duplicate removal:",
    before_duplicates
)

print(
    "Reviews after duplicate removal:",
    after_duplicates
)

print(
    "Duplicates removed:",
    before_duplicates - after_duplicates
)


# ============================================================
# 8. RATING CLEANING
# ============================================================

print("\n========== RATING CLEANING ==========")

before_rating_cleaning = reviews.count()

# Keep only ratings between 1 and 5
reviews = reviews.filter(
    (col("rating") >= 1) &
    (col("rating") <= 5)
)

after_rating_cleaning = reviews.count()

print(
    "Reviews before rating cleaning:",
    before_rating_cleaning
)

print(
    "Reviews after rating cleaning:",
    after_rating_cleaning
)

print(
    "Invalid ratings removed:",
    before_rating_cleaning - after_rating_cleaning
)


# ============================================================
# 9. HANDLE MISSING HELPFUL VOTES
# ============================================================

print("\n========== CLEANING HELPFUL VOTES ==========")

reviews = reviews.withColumn(
    "helpful_vote",
    when(
        col("helpful_vote").isNull(),
        0
    ).otherwise(
        col("helpful_vote")
    )
)

print("Missing helpful votes replaced with 0")


# ============================================================
# 10. REMOVE NULL / EMPTY REVIEW TEXT
# ============================================================

print("\n========== CLEANING REVIEW TEXT ==========")

before_text_cleaning = reviews.count()

reviews = reviews.filter(
    col("text").isNotNull() &
    (trim(col("text")) != "")
)

after_text_cleaning = reviews.count()

print(
    "Reviews before text cleaning:",
    before_text_cleaning
)

print(
    "Reviews after text cleaning:",
    after_text_cleaning
)

print(
    "Empty/null reviews removed:",
    before_text_cleaning - after_text_cleaning
)


# ============================================================
# 11. CREATE REVIEW LENGTH COLUMN
# ============================================================

print("\n========== CREATING REVIEW LENGTH ==========")

reviews = reviews.withColumn(
    "review_length",
    length(col("text"))
)

print("Column created: review_length")


# ============================================================
# 12. CONVERT TIMESTAMP TO DATE
# ============================================================

print("\n========== CREATING REVIEW DATE ==========")

reviews = reviews.withColumn(
    "review_date",
    from_unixtime(
        col("timestamp") / 1000
    ).cast("timestamp")
)

print("Column created: review_date")


# ============================================================
# 13. CREATE REVIEW YEAR
# ============================================================

reviews = reviews.withColumn(
    "review_year",
    year(col("review_date"))
)

print("Column created: review_year")


# ============================================================
# 14. CREATE REVIEW MONTH
# ============================================================

reviews = reviews.withColumn(
    "review_month",
    month(col("review_date"))
)

print("Column created: review_month")


# ============================================================
# 15. CHECK SPARK PARTITIONS
# ============================================================

print("\n========== PARTITIONS ==========")

number_of_partitions = reviews.rdd.getNumPartitions()

print(
    "Number of partitions:",
    number_of_partitions
)


# ============================================================
# 16. FINAL CLEANED DATA INFORMATION
# ============================================================

print("\n========== CLEANED DATA ==========")

final_count = reviews.count()

print(
    "Final number of reviews:",
    final_count
)

print(
    "Final number of columns:",
    len(reviews.columns)
)

print("\nFinal Schema:")

reviews.printSchema()


# ============================================================
# 17. DISPLAY CLEANED DATA
# ============================================================

print("\n========== SAMPLE CLEANED DATA ==========")

reviews.select(
    "asin",
    "rating",
    "helpful_vote",
    "text",
    "verified_purchase",
    "review_date",
    "review_year",
    "review_month",
    "review_length"
).show(
    10,
    truncate=False
)


# ============================================================
# 18. SAVE CLEANED DATA
# ============================================================
#
# IMPORTANT:
#
# We are NOT using:
#
# reviews.write.parquet(...)
#
# reviews.write.json(...)
#
# because those use Spark's Hadoop filesystem on Windows.
#
# Instead, we use Spark partitions and Python file handling.
#
# ============================================================

print("\n========== SAVING CLEANED DATA ==========")

output_dir = "output/clean_reviews"

# Create output folder using normal Python
os.makedirs(
    output_dir,
    exist_ok=True
)


# ------------------------------------------------------------
# Delete old output files if they exist
# ------------------------------------------------------------

for file_name in os.listdir(output_dir):

    file_path = os.path.join(
        output_dir,
        file_name
    )

    if os.path.isfile(file_path):

        os.remove(file_path)


# ------------------------------------------------------------
# Function to save each Spark partition
# ------------------------------------------------------------

def save_partition(partition_id, iterator):

    file_path = os.path.join(
        output_dir,
        f"part_{partition_id}.jsonl"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        for row in iterator:

            row_dict = row.asDict()

            file.write(
                json.dumps(
                    row_dict,
                    default=str
                )
                + "\n"
            )

    return [partition_id]


# ------------------------------------------------------------
# Save all Spark partitions
# ------------------------------------------------------------

partition_ids = reviews.rdd.mapPartitionsWithIndex(
    save_partition
).collect()


# ============================================================
# 19. VERIFY OUTPUT
# ============================================================

print("\n========== OUTPUT FILES ==========")

output_files = sorted(
    os.listdir(output_dir)
)

for file_name in output_files:

    print(
        file_name
    )


# ============================================================
# 20. FINAL RESULT
# ============================================================

print("\n========================================")
print("CLEANED DATASET SAVED SUCCESSFULLY")
print("========================================")

print(
    "Location:",
    output_dir
)

print(
    "Total reviews:",
    final_count
)

print(
    "Total partitions:",
    len(partition_ids)
)

print("========================================")


# ============================================================
# 21. STOP SPARK
# ============================================================

spark.stop()