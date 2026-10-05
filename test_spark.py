from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("AmazonReviewTest") \
    .master("local[*]") \
    .getOrCreate()

print("Spark version:", spark.version)

data = [
    ("Product A", 5),
    ("Product B", 4),
    ("Product C", 3)
]

df = spark.createDataFrame(
    data,
    ["product", "rating"]
)

df.show()

spark.stop()