# ============================================================
# PERSON 2 - SPARK PROCESSING + SPARK SQL
# Amazon Review Big Data Project
# ============================================================

import os
import sys
import csv
import json
import shutil

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    min,
    max,
    round,
    desc,
    asc,
    when
)

# ============================================================
# CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("AmazonReviews_Person2")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.hadoop.hadoop.native.lib", "false")
    .config("spark.hadoop.io.native.lib.available", "false")
    .config("spark.hadoop.fs.file.impl.disable.cache", "true")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_PATH = os.path.join(
    PROJECT_DIR,
    "output",
    "clean_reviews"
)

PROCESSED_PATH = os.path.join(
    PROJECT_DIR,
    "output",
    "processed_reviews"
)

SQL_RESULTS_PATH = os.path.join(
    PROJECT_DIR,
    "output",
    "spark_sql_results"
)

# ============================================================
# CLEAN OLD OUTPUT
# ============================================================

if os.path.exists(PROCESSED_PATH):
    shutil.rmtree(PROCESSED_PATH)

if os.path.exists(SQL_RESULTS_PATH):
    shutil.rmtree(SQL_RESULTS_PATH)

os.makedirs(
    PROCESSED_PATH,
    exist_ok=True
)

os.makedirs(
    SQL_RESULTS_PATH,
    exist_ok=True
)

# ============================================================
# TITLE
# ============================================================

print()
print("=" * 70)
print("PERSON 2 - SPARK PROCESSING + SPARK SQL")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Input path:")
print(INPUT_PATH)

# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT_PATH):

    print()
    print("ERROR: Input folder does not exist.")
    print(INPUT_PATH)

    spark.stop()
    sys.exit(1)

# ============================================================
# FIND JSONL FILES
# ============================================================

print()
print("[1] Searching for cleaned JSONL files...")
print("-" * 70)

json_files = []

for file_name in os.listdir(INPUT_PATH):

    if file_name.lower().endswith(".jsonl"):

        file_path = os.path.join(
            INPUT_PATH,
            file_name
        )

        if os.path.isfile(file_path):
            json_files.append(file_path)

json_files.sort()

if len(json_files) == 0:

    print()
    print("ERROR: No .jsonl files found.")
    print("Folder:", INPUT_PATH)

    spark.stop()
    sys.exit(1)

print()
print("JSONL files found:", len(json_files))

for file_path in json_files:
    print(" -", os.path.basename(file_path))

# ============================================================
# LOAD DATA
# ============================================================

print()
print("[2] Loading cleaned dataset...")
print("-" * 70)

try:

    df = (
        spark.read
        .option("multiLine", "false")
        .json(json_files)
    )

except Exception as e:

    print()
    print("ERROR WHILE READING JSON DATA")
    print("--------------------------------")
    print(e)

    spark.stop()
    sys.exit(1)

print()
print("Dataset loaded successfully.")

# ============================================================
# SCHEMA
# ============================================================

print()
print("[3] Schema")
print("-" * 70)

df.printSchema()

# ============================================================
# RECORD COUNT
# ============================================================

print()
print("[4] Total number of records")
print("-" * 70)

total_records = df.count()

print("Total records:", total_records)

# ============================================================
# SAMPLE
# ============================================================

print()
print("[5] Sample records")
print("-" * 70)

df.show(
    10,
    truncate=False
)

# ============================================================
# COLUMNS
# ============================================================

print()
print("[6] Dataset columns")
print("-" * 70)

for column_name in df.columns:
    print(" -", column_name)

print()
print("Number of columns:", len(df.columns))

# ============================================================
# SELECT TRANSFORMATION
# ============================================================

print()
print("[7] SELECT transformation")

possible_columns = [
    "asin",
    "parent_asin",
    "rating",
    "title",
    "text",
    "verified_purchase",
    "helpful_vote",
    "review_length",
    "year",
    "month"
]

available_columns = []

for column_name in possible_columns:

    if column_name in df.columns:
        available_columns.append(column_name)

if len(available_columns) > 0:

    selected_df = df.select(
        *available_columns
    )

    print()
    print("Selected columns:")
    print(available_columns)

    selected_df.show(
        10,
        truncate=False
    )

else:

    selected_df = df

    print("No expected columns found.")

# ============================================================
# FILTER TRANSFORMATION
# ============================================================

print()
print("[8] FILTER transformation")

if "rating" in df.columns:

    high_rating_df = df.filter(
        col("rating") >= 4
    )

    print()
    print("Reviews with rating >= 4:")

    high_rating_df.show(
        10,
        truncate=False
    )

    print(
        "Number of high-rated reviews:",
        high_rating_df.count()
    )

else:

    high_rating_df = None
    print("Rating column not available.")

# ============================================================
# RATING DISTRIBUTION
# ============================================================

print()
print("[9] Rating distribution")

if "rating" in df.columns:

    rating_distribution = (
        df
        .groupBy("rating")
        .agg(
            count("*").alias("review_count")
        )
        .orderBy(
            asc("rating")
        )
    )

    rating_distribution.show()

else:

    rating_distribution = None

# ============================================================
# RATING STATISTICS
# ============================================================

print()
print("[10] Rating statistics")

if "rating" in df.columns:

    rating_statistics = df.select(
        round(
            avg("rating"),
            2
        ).alias("average_rating"),

        min("rating").alias(
            "minimum_rating"
        ),

        max("rating").alias(
            "maximum_rating"
        )
    )

    rating_statistics.show()

else:

    rating_statistics = None

# ============================================================
# VERIFIED PURCHASE
# ============================================================

print()
print("[11] Verified purchase analysis")

if (
    "verified_purchase" in df.columns
    and "rating" in df.columns
):

    verified_analysis = (
        df
        .groupBy("verified_purchase")
        .agg(
            count("*").alias(
                "review_count"
            ),

            round(
                avg("rating"),
                2
            ).alias(
                "average_rating"
            )
        )
        .orderBy(
            desc("review_count")
        )
    )

    verified_analysis.show()

else:

    verified_analysis = None
    print(
        "verified_purchase or rating column missing."
    )

# ============================================================
# HELPFUL VOTES
# ============================================================

print()
print("[12] Helpful vote analysis")

if "helpful_vote" in df.columns:

    helpful_analysis = df.select(
        round(
            avg("helpful_vote"),
            2
        ).alias(
            "average_helpful_votes"
        ),

        min(
            "helpful_vote"
        ).alias(
            "minimum_helpful_votes"
        ),

        max(
            "helpful_vote"
        ).alias(
            "maximum_helpful_votes"
        )
    )

    helpful_analysis.show()

else:

    helpful_analysis = None
    print("helpful_vote column missing.")

# ============================================================
# PRODUCT COLUMN
# ============================================================

product_column = None

if "parent_asin" in df.columns:
    product_column = "parent_asin"

elif "asin" in df.columns:
    product_column = "asin"

# ============================================================
# PRODUCT ANALYSIS
# ============================================================

print()
print("[13] Product review analysis")

if (
    product_column is not None
    and "rating" in df.columns
):

    product_analysis = (
        df
        .groupBy(product_column)
        .agg(
            count("*").alias(
                "review_count"
            ),

            round(
                avg("rating"),
                2
            ).alias(
                "average_rating"
            ),

            min("rating").alias(
                "minimum_rating"
            ),

            max("rating").alias(
                "maximum_rating"
            )
        )
        .orderBy(
            desc("review_count")
        )
    )

    print()
    print("Top 20 products:")

    product_analysis.show(
        20,
        truncate=False
    )

else:

    product_analysis = None

    print(
        "Product ID or rating column missing."
    )

# ============================================================
# SPARK SQL
# ============================================================

print()
print("=" * 70)
print("SPARK SQL")
print("=" * 70)

# ============================================================
# TEMP VIEW
# ============================================================

print()
print("[14] Creating Spark SQL temporary view")

df.createOrReplaceTempView(
    "reviews"
)

print(
    "Temporary view 'reviews' created."
)

# ============================================================
# SQL TOTAL REVIEWS
# ============================================================

print()
print("[15] SQL - Total reviews")

sql_total_reviews = spark.sql("""
    SELECT
        COUNT(*) AS total_reviews
    FROM reviews
""")

sql_total_reviews.show()

# ============================================================
# SQL AVERAGE RATING
# ============================================================

if "rating" in df.columns:

    print()
    print("[16] SQL - Average rating")

    sql_average_rating = spark.sql("""
        SELECT
            ROUND(
                AVG(rating),
                2
            ) AS average_rating
        FROM reviews
    """)

    sql_average_rating.show()

else:

    sql_average_rating = None

# ============================================================
# SQL RATING DISTRIBUTION
# ============================================================

if "rating" in df.columns:

    print()
    print("[17] SQL - Rating distribution")

    sql_rating_distribution = spark.sql("""
        SELECT
            rating,
            COUNT(*) AS review_count
        FROM reviews
        GROUP BY rating
        ORDER BY rating ASC
    """)

    sql_rating_distribution.show()

else:

    sql_rating_distribution = None

# ============================================================
# SQL VERIFIED PURCHASE
# ============================================================

if (
    "verified_purchase" in df.columns
    and "rating" in df.columns
):

    print()
    print("[18] SQL - Verified purchase")

    sql_verified_purchase = spark.sql("""
        SELECT
            verified_purchase,
            COUNT(*) AS review_count,
            ROUND(
                AVG(rating),
                2
            ) AS average_rating
        FROM reviews
        GROUP BY verified_purchase
        ORDER BY review_count DESC
    """)

    sql_verified_purchase.show()

else:

    sql_verified_purchase = None

# ============================================================
# SQL HIGH RATINGS
# ============================================================

if "rating" in df.columns:

    print()
    print("[19] SQL - High rating reviews")

    sql_high_ratings = spark.sql("""
        SELECT
            rating,
            COUNT(*) AS review_count
        FROM reviews
        WHERE rating >= 4
        GROUP BY rating
        ORDER BY rating DESC
    """)

    sql_high_ratings.show()

else:

    sql_high_ratings = None

# ============================================================
# SQL LOW RATINGS
# ============================================================

if "rating" in df.columns:

    print()
    print("[20] SQL - Low rating reviews")

    sql_low_ratings = spark.sql("""
        SELECT
            rating,
            COUNT(*) AS review_count
        FROM reviews
        WHERE rating <= 2
        GROUP BY rating
        ORDER BY rating ASC
    """)

    sql_low_ratings.show()

else:

    sql_low_ratings = None

# ============================================================
# SQL PRODUCT ANALYSIS
# ============================================================

if (
    product_column is not None
    and "rating" in df.columns
):

    print()
    print("[21] SQL - Product analysis")

    sql_product_analysis = spark.sql(
        f"""
        SELECT
            {product_column} AS product_id,
            COUNT(*) AS review_count,
            ROUND(
                AVG(rating),
                2
            ) AS average_rating,
            MIN(rating) AS minimum_rating,
            MAX(rating) AS maximum_rating
        FROM reviews
        GROUP BY {product_column}
        ORDER BY review_count DESC
        LIMIT 100
        """
    )

    sql_product_analysis.show(
        20,
        truncate=False
    )

else:

    sql_product_analysis = None

# ============================================================
# SQL MOST HELPFUL REVIEWS
# ============================================================

if "helpful_vote" in df.columns:

    print()
    print("[22] SQL - Most helpful reviews")

    sql_helpful_reviews = spark.sql("""
        SELECT *
        FROM reviews
        ORDER BY helpful_vote DESC
        LIMIT 20
    """)

    sql_helpful_reviews.show(
        20,
        truncate=False
    )

else:

    sql_helpful_reviews = None

# ============================================================
# SQL REVIEW LENGTH
# ============================================================

if "review_length" in df.columns:

    print()
    print("[23] SQL - Review length")

    sql_review_length = spark.sql("""
        SELECT
            ROUND(
                AVG(review_length),
                2
            ) AS average_review_length,

            MIN(review_length)
                AS minimum_review_length,

            MAX(review_length)
                AS maximum_review_length

        FROM reviews
    """)

    sql_review_length.show()

else:

    sql_review_length = None

# ============================================================
# SQL REVIEW LENGTH BY RATING
# ============================================================

if (
    "review_length" in df.columns
    and "rating" in df.columns
):

    print()
    print("[24] SQL - Review length by rating")

    sql_length_rating = spark.sql("""
        SELECT
            rating,
            COUNT(*) AS review_count,
            ROUND(
                AVG(review_length),
                2
            ) AS average_review_length
        FROM reviews
        GROUP BY rating
        ORDER BY rating
    """)

    sql_length_rating.show()

else:

    sql_length_rating = None

# ============================================================
# SQL YEAR ANALYSIS
# ============================================================

if "year" in df.columns:

    print()
    print("[25] SQL - Reviews by year")

    sql_year_analysis = spark.sql("""
        SELECT
            year,
            COUNT(*) AS review_count,
            ROUND(
                AVG(rating),
                2
            ) AS average_rating
        FROM reviews
        GROUP BY year
        ORDER BY year
    """)

    sql_year_analysis.show()

else:

    sql_year_analysis = None

# ============================================================
# SQL MONTH ANALYSIS
# ============================================================

if "month" in df.columns:

    print()
    print("[26] SQL - Reviews by month")

    sql_month_analysis = spark.sql("""
        SELECT
            month,
            COUNT(*) AS review_count,
            ROUND(
                AVG(rating),
                2
            ) AS average_rating
        FROM reviews
        GROUP BY month
        ORDER BY month
    """)

    sql_month_analysis.show()

else:

    sql_month_analysis = None

# ============================================================
# SENTIMENT CLASSIFICATION
# ============================================================

print()
print("[27] Creating sentiment categories")

if "rating" in df.columns:

    processed_df = df.withColumn(
        "sentiment",

        when(
            col("rating") >= 4,
            "Positive"
        )

        .when(
            col("rating") == 3,
            "Neutral"
        )

        .otherwise(
            "Negative"
        )
    )

else:

    processed_df = df

# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

if "rating" in df.columns:

    print()
    print("[28] Sentiment analysis")

    sentiment_summary = (
        processed_df
        .groupBy("sentiment")
        .agg(
            count("*").alias(
                "review_count"
            )
        )
        .orderBy(
            desc("review_count")
        )
    )

    sentiment_summary.show()

else:

    sentiment_summary = None

# ============================================================
# PYTHON VALUE CONVERTER
# ============================================================

def convert_value(value):

    if value is None:
        return ""

    if isinstance(
        value,
        (list, tuple, dict)
    ):

        try:

            return json.dumps(
                value,
                ensure_ascii=False,
                default=str
            )

        except Exception:

            return str(value)

    return value

# ============================================================
# SAVE SMALL DATAFRAME RESULT
# ============================================================

def save_dataframe_csv(
    dataframe,
    folder_name
):

    if dataframe is None:
        return

    output_folder = os.path.join(
        SQL_RESULTS_PATH,
        folder_name
    )

    try:

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        output_file = os.path.join(
            output_folder,
            "result.csv"
        )

        columns = dataframe.columns

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                columns
            )

            for row in dataframe.toLocalIterator():

                values = []

                for column_name in columns:

                    value = row[column_name]

                    values.append(
                        convert_value(value)
                    )

                writer.writerow(
                    values
                )

        print(
            "Saved:",
            output_file
        )

    except Exception as e:

        print()
        print(
            "Could not save",
            folder_name,
            ":",
            e
        )

# ============================================================
# SAVE PROCESSED DATASET
# IMPORTANT:
# DO NOT TRANSFER 3 MILLION FULL REVIEWS TO PYTHON.
# ONLY SAVE A 10,000-ROW PROCESSED SUMMARY.
# ============================================================

def save_processed_summary(dataframe):

    if dataframe is None:
        return

    try:

        output_file = os.path.join(
            PROCESSED_PATH,
            "processed_reviews_summary.csv"
        )

        required_columns = [
            "asin",
            "parent_asin",
            "rating",
            "verified_purchase",
            "helpful_vote",
            "review_length",
            "year",
            "month",
            "sentiment"
        ]

        summary_columns = []

        for column_name in required_columns:

            if column_name in dataframe.columns:

                summary_columns.append(
                    column_name
                )

        if len(summary_columns) == 0:

            print(
                "No suitable columns available "
                "for processed summary."
            )

            return

        summary_df = dataframe.select(
            *summary_columns
        )

        # Only 10,000 rows are transferred to Python.
        sample_df = summary_df.limit(10000)

        rows = sample_df.collect()

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                summary_columns
            )

            for row in rows:

                values = []

                for column_name in summary_columns:

                    values.append(
                        convert_value(
                            row[column_name]
                        )
                    )

                writer.writerow(values)

        print()
        print(
            "Processed summary saved to:"
        )
        print(output_file)

        print(
            "Rows saved:",
            len(rows)
        )

    except Exception as e:

        print()
        print(
            "Could not save processed summary:"
        )
        print(e)

# ============================================================
# SAVE PROCESSED SUMMARY
# ============================================================

print()
print("[29] Saving processed dataset")
print("-" * 70)

save_processed_summary(
    processed_df
)

# ============================================================
# SAVE SQL RESULTS
# ============================================================

print()
print("[30] Saving SQL results")
print("-" * 70)

save_dataframe_csv(
    rating_distribution,
    "rating_distribution"
)

save_dataframe_csv(
    rating_statistics,
    "rating_statistics"
)

save_dataframe_csv(
    verified_analysis,
    "verified_purchase"
)

save_dataframe_csv(
    helpful_analysis,
    "helpful_votes"
)

save_dataframe_csv(
    product_analysis,
    "product_analysis"
)

save_dataframe_csv(
    sql_total_reviews,
    "sql_total_reviews"
)

save_dataframe_csv(
    sql_average_rating,
    "sql_average_rating"
)

save_dataframe_csv(
    sql_rating_distribution,
    "sql_rating_distribution"
)

save_dataframe_csv(
    sql_verified_purchase,
    "sql_verified_purchase"
)

save_dataframe_csv(
    sql_high_ratings,
    "sql_high_ratings"
)

save_dataframe_csv(
    sql_low_ratings,
    "sql_low_ratings"
)

save_dataframe_csv(
    sql_product_analysis,
    "sql_product_analysis"
)

save_dataframe_csv(
    sql_helpful_reviews,
    "sql_helpful_reviews"
)

save_dataframe_csv(
    sql_review_length,
    "sql_review_length"
)

save_dataframe_csv(
    sql_length_rating,
    "sql_length_rating"
)

save_dataframe_csv(
    sql_year_analysis,
    "sql_year_analysis"
)

save_dataframe_csv(
    sql_month_analysis,
    "sql_month_analysis"
)

save_dataframe_csv(
    sentiment_summary,
    "sentiment_summary"
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PERSON 2 COMPLETED")
print("=" * 70)

print()
print("Input:")
print(INPUT_PATH)

print()
print("Processed output:")
print(PROCESSED_PATH)

print()
print("SQL results:")
print(SQL_RESULTS_PATH)

print()
print("Operations completed:")

print("1. Loaded cleaned Amazon review data")
print("2. Displayed schema")
print("3. Displayed sample records")
print("4. SELECT transformation")
print("5. FILTER transformation")
print("6. Rating distribution")
print("7. Rating statistics")
print("8. Verified purchase analysis")
print("9. Helpful vote analysis")
print("10. Product analysis")
print("11. Spark SQL temporary view")
print("12. SQL total reviews")
print("13. SQL average rating")
print("14. SQL rating distribution")
print("15. SQL verified purchase analysis")
print("16. SQL high-rating analysis")
print("17. SQL low-rating analysis")
print("18. SQL product analysis")
print("19. SQL helpful review analysis")
print("20. SQL review length analysis")
print("21. SQL review length by rating")
print("22. SQL year analysis")
print("23. SQL month analysis")
print("24. Sentiment classification")
print("25. Sentiment analysis")
print("26. Saved processed summary")
print("27. Saved SQL results")

print()
print("Person 2 work is complete.")
print("The output can now be used by Person 3.")

print()
print("=" * 70)

# ============================================================
# STOP SPARK
# ============================================================

spark.stop()

print()
print("Spark session stopped.")