# ============================================================
# PERSON 3 - PRODUCT + REVIEW ANALYSIS
# Amazon Review Big Data Project
# ============================================================

import os
import sys
import csv
import shutil
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    desc,
    asc,
    round,
    min,
    max,
    corr,
    when,
    lower,
    lit,
    to_timestamp,
    from_unixtime,
    year as year_function,
    month as month_function
)


# ============================================================
# CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("AmazonReviews_Person3")
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

OUTPUT_PATH = os.path.join(
    PROJECT_DIR,
    "output"
)

CLEAN_REVIEWS_PATH = os.path.join(
    OUTPUT_PATH,
    "clean_reviews"
)

PROCESSED_REVIEWS_PATH = os.path.join(
    OUTPUT_PATH,
    "processed_reviews"
)

PRODUCT_ANALYSIS_PATH = os.path.join(
    OUTPUT_PATH,
    "product_analysis"
)

REVIEW_ANALYSIS_PATH = os.path.join(
    OUTPUT_PATH,
    "review_analysis"
)

ENRICHED_REVIEWS_PATH = os.path.join(
    OUTPUT_PATH,
    "enriched_reviews"
)


# ============================================================
# CLEAN PERSON 3 OUTPUT
# ============================================================

for folder in [
    PRODUCT_ANALYSIS_PATH,
    REVIEW_ANALYSIS_PATH,
    ENRICHED_REVIEWS_PATH
]:

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(
        folder,
        exist_ok=True
    )


# ============================================================
# TITLE
# ============================================================

print()
print("=" * 70)
print("PERSON 3 - PRODUCT + REVIEW ANALYSIS")
print("=" * 70)

print()
print("Project directory:")
print(PROJECT_DIR)

print()
print("Output directory:")
print(OUTPUT_PATH)


# ============================================================
# FIND INPUT DATA
# ============================================================

print()
print("[1] Finding Person 2 data...")
print("-" * 70)

json_files = []

if os.path.exists(CLEAN_REVIEWS_PATH):

    for file_name in os.listdir(CLEAN_REVIEWS_PATH):

        if file_name.lower().endswith(".jsonl"):

            file_path = os.path.join(
                CLEAN_REVIEWS_PATH,
                file_name
            )

            if os.path.isfile(file_path):
                json_files.append(file_path)


json_files.sort()


# ============================================================
# LOAD FULL CLEANED DATA WHEN AVAILABLE
# ============================================================

if len(json_files) > 0:

    print()
    print("Using full cleaned review data from:")
    print(CLEAN_REVIEWS_PATH)

    print()
    print("JSONL files found:", len(json_files))

    for file_path in json_files:
        print(
            " -",
            os.path.basename(file_path)
        )

    try:

        reviews = (
            spark.read
            .option(
                "multiLine",
                "false"
            )
            .json(json_files)
        )

    except Exception as e:

        print()
        print("ERROR while reading cleaned reviews:")
        print(e)

        spark.stop()
        sys.exit(1)

else:

    # ========================================================
    # FALLBACK TO PERSON 2 PROCESSED SUMMARY
    # ========================================================

    summary_file = os.path.join(
        PROCESSED_REVIEWS_PATH,
        "processed_reviews_summary.csv"
    )

    if not os.path.exists(summary_file):

        print()
        print("ERROR: No Person 2 review data found.")

        print()
        print("Expected either:")

        print(
            CLEAN_REVIEWS_PATH
            + "/*.jsonl"
        )

        print(
            "or:"
        )

        print(
            summary_file
        )

        spark.stop()
        sys.exit(1)

    print()
    print(
        "Full clean_reviews not found."
    )

    print(
        "Using Person 2 processed summary:"
    )

    print(summary_file)

    try:

        reviews = (
            spark.read
            .option(
                "header",
                "true"
            )
            .option(
                "inferSchema",
                "true"
            )
            .csv(summary_file)
        )

    except Exception as e:

        print()
        print(
            "ERROR while reading processed summary:"
        )

        print(e)

        spark.stop()
        sys.exit(1)


# ============================================================
# DATASET INFORMATION
# ============================================================

print()
print("[2] Dataset loaded successfully")
print("-" * 70)

print()
print("Columns:")

for column_name in reviews.columns:

    print(
        " -",
        column_name
    )

print()
print(
    "Number of columns:",
    len(reviews.columns)
)


# ============================================================
# RECORD COUNT
# ============================================================

total_reviews = reviews.count()

print()
print(
    "Total reviews:",
    total_reviews
)


# ============================================================
# PREPARE REVIEW LENGTH
# ============================================================

if "review_length" not in reviews.columns:

    if "text" in reviews.columns:

        reviews = reviews.withColumn(
            "review_length",
            when(
                col("text").isNotNull(),
                col("text").cast("string")
            ).otherwise(
                lit("")
            )
        )

        reviews = reviews.withColumn(
            "review_length",
            col("review_length").alias(
                "review_length"
            )
        )

        # Calculate actual character length
        from pyspark.sql.functions import length

        reviews = reviews.withColumn(
            "review_length",
            length(
                col("review_length")
            )
        )


# ============================================================
# PREPARE TIMESTAMP / YEAR / MONTH
# ============================================================

columns = reviews.columns

if (
    "year" not in columns
    or "month" not in columns
):

    if "timestamp" in columns:

        timestamp_type = dict(
            reviews.dtypes
        ).get(
            "timestamp"
        )

        if timestamp_type in [
            "bigint",
            "long",
            "int",
            "double",
            "float"
        ]:

            timestamp_column = from_unixtime(
                (
                    col("timestamp")
                    .cast("double")
                    / 1000
                ).cast("long")
            )

        else:

            timestamp_column = to_timestamp(
                col("timestamp")
            )

        reviews = reviews.withColumn(
            "_review_timestamp",
            timestamp_column
        )

        if "year" not in reviews.columns:

            reviews = reviews.withColumn(
                "year",
                year_function(
                    col("_review_timestamp")
                )
            )

        if "month" not in reviews.columns:

            reviews = reviews.withColumn(
                "month",
                month_function(
                    col("_review_timestamp")
                )
            )

else:

    reviews = reviews.withColumn(
        "year",
        col("year").cast("int")
    )

    reviews = reviews.withColumn(
        "month",
        col("month").cast("int")
    )


# ============================================================
# PREPARE HELPFUL VOTE
# ============================================================

if "helpful_vote" in reviews.columns:

    reviews = reviews.withColumn(
        "helpful_vote",
        col("helpful_vote").cast("double")
    )


# ============================================================
# TEMPORARY SQL VIEW
# ============================================================

reviews.createOrReplaceTempView(
    "reviews"
)


# ============================================================
# HELPER FUNCTION - SAVE SPARK DATAFRAME TO CSV
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


def save_dataframe_csv(
    dataframe,
    folder_path,
    file_name="result.csv"
):

    if dataframe is None:
        return

    try:

        os.makedirs(
            folder_path,
            exist_ok=True
        )

        output_file = os.path.join(
            folder_path,
            file_name
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

                    values.append(
                        convert_value(
                            row[column_name]
                        )
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
            "Could not save:",
            folder_path
        )

        print(e)


# ============================================================
# 1. PRODUCT POPULARITY
# ============================================================

print()
print("=" * 70)
print("1. PRODUCT POPULARITY")
print("=" * 70)

product_column = None

if "asin" in reviews.columns:

    product_column = "asin"

elif "parent_asin" in reviews.columns:

    product_column = "parent_asin"


if product_column is None:

    print(
        "ERROR: No product ID column found."
    )

    product_popularity = None

else:

    product_popularity = (
        reviews
        .groupBy(product_column)
        .agg(
            count("*").alias(
                "review_count"
            )
        )
        .orderBy(
            desc("review_count")
        )
    )

    print()
    print(
        "Top 20 most reviewed products:"
    )

    product_popularity.show(
        20,
        truncate=False
    )

    save_dataframe_csv(
        product_popularity,
        PRODUCT_ANALYSIS_PATH,
        "product_popularity.csv"
    )


# ============================================================
# PRODUCT POPULARITY + AVERAGE RATING
# ============================================================

if (
    product_column is not None
    and "rating" in reviews.columns
):

    product_performance = (
        reviews
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

            round(
                avg("helpful_vote"),
                2
            ).alias(
                "average_helpful_votes"
            )

            if "helpful_vote" in reviews.columns
            else count("*").alias(
                "_dummy"
            )
        )
        .orderBy(
            desc("review_count")
        )
    )

    if "_dummy" in product_performance.columns:

        product_performance = (
            product_performance
            .drop("_dummy")
        )

    save_dataframe_csv(
        product_performance,
        PRODUCT_ANALYSIS_PATH,
        "product_performance.csv"
    )


# ============================================================
# 2. MOST HELPFUL REVIEWS
# ============================================================

print()
print("=" * 70)
print("2. MOST HELPFUL REVIEWS")
print("=" * 70)

if "helpful_vote" in reviews.columns:

    helpful_columns = []

    preferred_columns = [
        "asin",
        "parent_asin",
        "rating",
        "helpful_vote",
        "verified_purchase",
        "review_length",
        "title",
        "text"
    ]

    for column_name in preferred_columns:

        if column_name in reviews.columns:

            helpful_columns.append(
                column_name
            )

    helpful_reviews = (
        reviews
        .select(
            *helpful_columns
        )
        .orderBy(
            col("helpful_vote").desc()
        )
        .limit(20)
    )

    print()
    print(
        "Top 20 most helpful reviews:"
    )

    helpful_reviews.show(
        20,
        truncate=False
    )

    save_dataframe_csv(
        helpful_reviews,
        REVIEW_ANALYSIS_PATH,
        "most_helpful_reviews.csv"
    )

else:

    helpful_reviews = None

    print(
        "helpful_vote column not available."
    )


# ============================================================
# 3. VERIFIED VS NON-VERIFIED CUSTOMERS
# ============================================================

print()
print("=" * 70)
print("3. VERIFIED VS NON-VERIFIED CUSTOMERS")
print("=" * 70)

if "verified_purchase" in reviews.columns:

    verified_reviews = reviews.withColumn(
        "customer_type",
        when(
            lower(
                col("verified_purchase").cast("string")
            ) == "true",
            "Verified Purchase"
        ).otherwise(
            "Non-Verified Purchase"
        )
    )

    aggregation_list = [
        count("*").alias(
            "review_count"
        )
    ]

    if "rating" in verified_reviews.columns:

        aggregation_list.append(
            round(
                avg("rating"),
                2
            ).alias(
                "average_rating"
            )
        )

    if "helpful_vote" in verified_reviews.columns:

        aggregation_list.append(
            round(
                avg("helpful_vote"),
                2
            ).alias(
                "average_helpful_votes"
            )
        )

    verified_analysis = (
        verified_reviews
        .groupBy(
            "customer_type"
        )
        .agg(
            *aggregation_list
        )
        .orderBy(
            desc("review_count")
        )
    )

    print()
    print(
        "Verified vs Non-Verified analysis:"
    )

    verified_analysis.show(
        truncate=False
    )

    save_dataframe_csv(
        verified_analysis,
        REVIEW_ANALYSIS_PATH,
        "verified_vs_nonverified.csv"
    )

else:

    verified_analysis = None

    print(
        "verified_purchase column not available."
    )


# ============================================================
# 4. REVIEW TRENDS BY YEAR
# ============================================================

print()
print("=" * 70)
print("4. REVIEW TRENDS")
print("=" * 70)

if "year" in reviews.columns:

    yearly_trends = (
        reviews
        .groupBy("year")
        .agg(
            count("*").alias(
                "review_count"
            )
        )
        .orderBy(
            asc("year")
        )
    )

    print()
    print(
        "Review activity by year:"
    )

    yearly_trends.show(
        100,
        truncate=False
    )

    save_dataframe_csv(
        yearly_trends,
        REVIEW_ANALYSIS_PATH,
        "review_trends_year.csv"
    )

else:

    yearly_trends = None

    print(
        "year column not available."
    )


# ============================================================
# REVIEW TRENDS BY MONTH
# ============================================================

if "month" in reviews.columns:

    monthly_trends = (
        reviews
        .groupBy("month")
        .agg(
            count("*").alias(
                "review_count"
            )
        )
        .orderBy(
            asc("month")
        )
    )

    print()
    print(
        "Review activity by month:"
    )

    monthly_trends.show(
        20,
        truncate=False
    )

    save_dataframe_csv(
        monthly_trends,
        REVIEW_ANALYSIS_PATH,
        "review_trends_month.csv"
    )

else:

    monthly_trends = None


# ============================================================
# YEAR + MONTH TREND
# ============================================================

if (
    "year" in reviews.columns
    and "month" in reviews.columns
):

    year_month_trends = (
        reviews
        .groupBy(
            "year",
            "month"
        )
        .agg(
            count("*").alias(
                "review_count"
            )
        )
        .orderBy(
            asc("year"),
            asc("month")
        )
    )

    save_dataframe_csv(
        year_month_trends,
        REVIEW_ANALYSIS_PATH,
        "review_trends_year_month.csv"
    )


# ============================================================
# 5. REVIEW LENGTH ANALYSIS
# ============================================================

print()
print("=" * 70)
print("5. REVIEW LENGTH ANALYSIS")
print("=" * 70)

if (
    "review_length" in reviews.columns
    and "helpful_vote" in reviews.columns
):

    review_length_analysis = reviews.select(
        round(
            avg("review_length"),
            2
        ).alias(
            "average_review_length"
        ),

        min(
            "review_length"
        ).alias(
            "minimum_review_length"
        ),

        max(
            "review_length"
        ).alias(
            "maximum_review_length"
        ),

        round(
            avg("helpful_vote"),
            2
        ).alias(
            "average_helpful_votes"
        ),

        round(
            corr(
                "review_length",
                "helpful_vote"
            ),
            4
        ).alias(
            "length_helpfulness_correlation"
        )
    )

    print()
    print(
        "Review length analysis:"
    )

    review_length_analysis.show(
        truncate=False
    )

    save_dataframe_csv(
        review_length_analysis,
        REVIEW_ANALYSIS_PATH,
        "review_length_analysis.csv"
    )


    # ========================================================
    # REVIEW LENGTH BUCKET ANALYSIS
    # ========================================================

    review_length_buckets = (
        reviews
        .withColumn(
            "review_length_category",

            when(
                col("review_length") < 50,
                "Very Short (<50)"
            )

            .when(
                col("review_length") < 200,
                "Short (50-199)"
            )

            .when(
                col("review_length") < 500,
                "Medium (200-499)"
            )

            .when(
                col("review_length") < 1000,
                "Long (500-999)"
            )

            .otherwise(
                "Very Long (1000+)"
            )
        )
        .groupBy(
            "review_length_category"
        )
        .agg(

            count("*").alias(
                "review_count"
            ),

            round(
                avg("helpful_vote"),
                2
            ).alias(
                "average_helpful_votes"
            ),

            round(
                avg("rating"),
                2
            ).alias(
                "average_rating"
            )
        )
    )

    print()
    print(
        "Helpful votes by review length:"
    )

    review_length_buckets.show(
        truncate=False
    )

    save_dataframe_csv(
        review_length_buckets,
        REVIEW_ANALYSIS_PATH,
        "helpfulness_by_review_length.csv"
    )

else:

    review_length_analysis = None
    review_length_buckets = None

    print(
        "review_length or helpful_vote column unavailable."
    )


# ============================================================
# SPARK SQL - PRODUCT POPULARITY
# ============================================================

print()
print("=" * 70)
print("SPARK SQL ANALYSIS")
print("=" * 70)

if product_column is not None:

    sql_product_popularity = spark.sql(
        f"""
        SELECT
            {product_column} AS product_id,
            COUNT(*) AS review_count
        FROM reviews
        GROUP BY {product_column}
        ORDER BY review_count DESC
        LIMIT 100
        """
    )

    print()
    print(
        "Spark SQL - Top 100 products:"
    )

    sql_product_popularity.show(
        20,
        truncate=False
    )

    save_dataframe_csv(
        sql_product_popularity,
        PRODUCT_ANALYSIS_PATH,
        "sql_top_products.csv"
    )


# ============================================================
# SPARK SQL - VERIFIED PURCHASE
# ============================================================

if "verified_purchase" in reviews.columns:

    sql_verified_analysis = spark.sql("""
        SELECT
            CASE
                WHEN LOWER(
                    CAST(verified_purchase AS STRING)
                ) = 'true'
                THEN 'Verified Purchase'
                ELSE 'Non-Verified Purchase'
            END AS customer_type,

            COUNT(*) AS review_count,

            ROUND(
                AVG(rating),
                2
            ) AS average_rating,

            ROUND(
                AVG(helpful_vote),
                2
            ) AS average_helpful_votes

        FROM reviews

        GROUP BY
            CASE
                WHEN LOWER(
                    CAST(verified_purchase AS STRING)
                ) = 'true'
                THEN 'Verified Purchase'
                ELSE 'Non-Verified Purchase'
            END

        ORDER BY review_count DESC
    """)

    print()
    print(
        "Spark SQL - Verified vs Non-Verified:"
    )

    sql_verified_analysis.show(
        truncate=False
    )

    save_dataframe_csv(
        sql_verified_analysis,
        REVIEW_ANALYSIS_PATH,
        "sql_verified_analysis.csv"
    )


# ============================================================
# PRODUCT METADATA JOIN
# ============================================================
#
# Metadata is OPTIONAL.
#
# Person 3 tasks 1-5 do NOT require product metadata.
#
# Since metadata is not available, this section safely skips
# the join and creates an information file.
#
# Later, when actual product metadata is available, this
# section can be updated after checking its real schema.
# ============================================================

print()
print("=" * 70)
print("6. PRODUCT METADATA JOIN")
print("=" * 70)

metadata_candidates = [

    os.path.join(
        PROJECT_DIR,
        "data",
        "product_metadata.jsonl"
    ),

    os.path.join(
        PROJECT_DIR,
        "data",
        "product_metadata.json"
    ),

    os.path.join(
        PROJECT_DIR,
        "data",
        "product_metadata.csv"
    )

]

metadata_path = None

for candidate in metadata_candidates:

    if os.path.exists(candidate):

        metadata_path = candidate
        break


if metadata_path is None:

    print()
    print(
        "Product metadata not found."
    )

    print(
        "Skipping metadata JOIN."
    )

    print()
    print(
        "Tasks 1-5 are complete without metadata."
    )

    metadata_note = os.path.join(
        ENRICHED_REVIEWS_PATH,
        "README.txt"
    )

    with open(
        metadata_note,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "Product metadata was not available.\n"
        )

        file.write(
            "Therefore the product metadata JOIN "
            "was skipped.\n"
        )

        file.write(
            "Person 3 Tasks 1-5 were completed "
            "using review data.\n"
        )

        file.write(
            "Add the actual Amazon product metadata "
            "file later to perform the JOIN.\n"
        )

else:

    print()
    print(
        "Product metadata found:"
    )

    print(metadata_path)

    print()
    print(
        "Metadata JOIN should be configured only "
        "after checking the actual metadata schema."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PERSON 3 COMPLETED")
print("=" * 70)

print()
print("Completed analyses:")

print("1. Product popularity")
print("2. Most helpful reviews")
print("3. Verified vs non-verified customers")
print("4. Review trends")
print("5. Review length and helpfulness analysis")

print()
print(
    "6. Product metadata JOIN:"
)

if metadata_path is None:

    print(
        "Skipped - metadata not available"
    )

else:

    print(
        "Metadata found - JOIN configuration required"
    )

print()
print("Output folders:")

print(
    "Product analysis:",
    PRODUCT_ANALYSIS_PATH
)

print(
    "Review analysis:",
    REVIEW_ANALYSIS_PATH
)

print(
    "Enriched reviews:",
    ENRICHED_REVIEWS_PATH
)

print()
print(
    "Person 3 work is complete."
)

print(
    "The output can now be used by Person 4."
)

print()
print("=" * 70)


# ============================================================
# STOP SPARK
# ============================================================

spark.stop()

print()
print("Spark session stopped.")