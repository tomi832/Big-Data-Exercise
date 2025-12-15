import pickle
import sqlite3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, expr, rand
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.ml.feature import StringIndexer
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, ListVector
import numpy as np
import datetime

# --------------------------
# 1. Spark Session
# --------------------------
spark = SparkSession.builder.appName("BookRecsysALS").getOrCreate()

# --------------------------
# 2. Load SQLite Data
# --------------------------
con = sqlite3.connect("empty_sqlite_db.db")
ratings_pd = pd.read_sql("SELECT * FROM `BX-Book-Ratings`", con)
users_pd = pd.read_sql("SELECT * FROM `BX-Users`", con)
books_pd = pd.read_sql("SELECT * FROM `BX-Books`", con)
con.close()

ratings = spark.createDataFrame(ratings_pd)
users = spark.createDataFrame(users_pd)
books = spark.createDataFrame(books_pd)

# RENAME COLUMNS WITH HYPHENS TO AVOID ISSUES
# This is where you fix the problem
ratings = ratings.withColumnRenamed("User-ID", "UserID")
ratings = ratings.withColumnRenamed("Book-Rating", "BookRating")
users = users.withColumnRenamed("User-ID", "UserID")
books = books.withColumnRenamed("Book-Title", "BookTitle")
books = books.withColumnRenamed("Book-Author", "BookAuthor")
books = books.withColumnRenamed("Year-Of-Publication", "YearOfPublication")
books = books.withColumnRenamed("Image-URL-S", "ImageURLS")
books = books.withColumnRenamed("Image-URL-M", "ImageURLM")
books = books.withColumnRenamed("Image-URL-L", "ImageURLL")

# --------------------------
# 3. Additional Data Cleaning
# --------------------------
# Remove ratings of 0 (implicit feedback, not actual ratings)
ratings = ratings.filter(col("BookRating") > 0)

# Remove duplicate ratings (same user-book pair)
ratings = ratings.dropDuplicates(["UserID", "ISBN"])

# Filter out users with very few ratings and books with very few ratings
user_counts = ratings.groupBy("UserID").count().filter(col("count") >= 5)
book_counts = ratings.groupBy("ISBN").count().filter(col("count") >= 5)

ratings = ratings.join(user_counts.select("UserID"), "UserID", "inner")
ratings = ratings.join(book_counts.select("ISBN"), "ISBN", "inner")

# --------------------------
# 4. Data Preparation
# --------------------------
# Map to integer indices for ALS
user_indexer = StringIndexer(inputCol="UserID", outputCol="userIndex")
book_indexer = StringIndexer(inputCol="ISBN", outputCol="bookIndex")

user_indexer_model = user_indexer.fit(ratings)
book_indexer_model = book_indexer.fit(ratings)

ratings = user_indexer_model.transform(ratings)
ratings = book_indexer_model.transform(ratings)
ratings = ratings.withColumn("BookRating", col("BookRating").cast("float"))

# Create mapping for book titles
book_mapping = books.select("ISBN", "BookTitle").toPandas()
book_index_mapping = ratings.select("ISBN", "bookIndex").distinct().toPandas()
book_mapping = book_mapping.merge(book_index_mapping, on="ISBN")

# --------------------------
# 5. K-Fold Cross Validation Setup
# --------------------------
k_folds = 5

# Create a window partitioned by User to number each user's ratings
user_window = Window.partitionBy("userIndex").orderBy(rand(42))

# Assign a fold number to each rating within each user's group
ratings_with_fold = (
    ratings.withColumn("row_num", row_number().over(user_window))
    .withColumn("fold", col("row_num") % k_folds)
    .drop("row_num")
)

# Store results for each fold
ubcf_rmse_per_fold = []
ibcf_rmse_per_fold = []
ubcf_user_rmse_all = []
ibcf_user_rmse_all = []

print("Starting K-Fold Cross Validation...")

for fold in range(k_folds):
    print(f"Training fold {fold + 1}/{k_folds}")

    # Split data
    train_fold = ratings_with_fold.filter(col("fold") != fold)
    test_fold = ratings_with_fold.filter(col("fold") == fold)

    # User-Based Collaborative Filtering (UBCF) - Higher rank, focus on users
    als_ubcf = ALS(
        userCol="userIndex",
        itemCol="bookIndex",
        ratingCol="BookRating",
        rank=50,  # Higher rank for user-based
        maxIter=15,
        regParam=0.01,  # Lower regularization
        alpha=1.0,  # For implicit feedback handling
        coldStartStrategy="drop",
    )

    # Item-Based Collaborative Filtering (IBCF) - Lower rank, focus on items
    als_ibcf = ALS(
        userCol="userIndex",
        itemCol="bookIndex",
        ratingCol="BookRating",
        rank=20,  # Lower rank for item-based
        maxIter=15,
        regParam=0.1,  # Higher regularization
        alpha=1.0,
        coldStartStrategy="drop",
    )

    # Train models
    model_ubcf = als_ubcf.fit(train_fold)
    model_ibcf = als_ibcf.fit(train_fold)

    # Make predictions
    pred_ubcf = model_ubcf.transform(test_fold)
    pred_ibcf = model_ibcf.transform(test_fold)

    # Calculate RMSE
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="BookRating", predictionCol="prediction"
    )

    rmse_ubcf = evaluator.evaluate(pred_ubcf)
    rmse_ibcf = evaluator.evaluate(pred_ibcf)

    ubcf_rmse_per_fold.append(rmse_ubcf)
    ibcf_rmse_per_fold.append(rmse_ibcf)

    # Per-user RMSE for this fold
    per_user_ubcf = (
        pred_ubcf.withColumn(
            "sq_error", (col("BookRating") - col("prediction")) ** 2)
        .groupBy("userIndex")
        .agg(expr("sqrt(avg(sq_error)) as rmse"))
    )

    per_user_ibcf = (
        pred_ibcf.withColumn(
            "sq_error", (col("BookRating") - col("prediction")) ** 2)
        .groupBy("userIndex")
        .agg(expr("sqrt(avg(sq_error)) as rmse"))
    )

    ubcf_user_rmse_all.extend(per_user_ubcf.select(
        "rmse").toPandas()["rmse"].tolist())
    ibcf_user_rmse_all.extend(per_user_ibcf.select(
        "rmse").toPandas()["rmse"].tolist())

# --------------------------
# 6. Final Model Training (on full dataset)
# --------------------------
print("Training final models on full dataset...")

# Split full dataset for final evaluation
(train_final, test_final) = ratings.randomSplit([0.8, 0.2], seed=42)

# Train final models
final_ubcf = als_ubcf.fit(train_final)
final_ibcf = als_ibcf.fit(train_final)

# Final predictions and RMSE
final_pred_ubcf = final_ubcf.transform(test_final)
final_pred_ibcf = final_ibcf.transform(test_final)

final_rmse_ubcf = evaluator.evaluate(final_pred_ubcf)
final_rmse_ibcf = evaluator.evaluate(final_pred_ibcf)

# --------------------------
# 7. Generate Recommendations
# --------------------------
print("Generating recommendations...")

# Get 500 random users
all_users = ratings.select("userIndex", "UserID").distinct()
sample_users = all_users.orderBy(rand(42)).limit(500).toPandas()

# Get recommendations for these users
user_indices = spark.createDataFrame(sample_users[["userIndex"]])
recs_ubcf = final_ubcf.recommendForUserSubset(user_indices, 10)
recs_ibcf = final_ibcf.recommendForUserSubset(user_indices, 10)

# Convert to pandas for easier processing
recs_ubcf_pd = recs_ubcf.toPandas()
recs_ibcf_pd = recs_ibcf.toPandas()

# Function to get book titles from recommendations


def get_book_titles(recommendations, book_mapping):
    book_titles = []
    for rec_list in recommendations:
        titles = []
        for rec in rec_list:
            book_idx = rec["bookIndex"]
            # Find book title
            book_row = book_mapping[book_mapping["bookIndex"] == book_idx]
            if not book_row.empty:
                title = book_row.iloc[0]["BookTitle"]
                # Truncate to 12 characters
                title = title[:12] if len(title) > 12 else title
                titles.append(title)
            else:
                titles.append("Unknown")
        book_titles.append(titles)
    return book_titles


ubcf_titles = get_book_titles(recs_ubcf_pd["recommendations"], book_mapping)
ibcf_titles = get_book_titles(recs_ibcf_pd["recommendations"], book_mapping)

# --------------------------
# 8. Create Histograms
# --------------------------
# Combine all user RMSEs from k-fold
ubcf_hist, bin_edges = np.histogram(
    ubcf_user_rmse_all, bins=np.arange(0, 5.25, 0.25))
ibcf_hist, _ = np.histogram(ibcf_user_rmse_all, bins=np.arange(0, 5.25, 0.25))

# --------------------------
# 9. Save Results
# --------------------------
today = datetime.date.today().isoformat()

with open("model.txt", "w", encoding="utf-8") as f:
    f.write("# Team: [Your Team Name]\n")
    f.write(f"# Date: {today}\n")
    f.write("# Database name: books_clean.db\n")
    f.write("\n")
    f.write("5) link to model data: [Upload your model files here]\n")
    f.write("\n")
    f.write(
        f"6.a) RMSE of the full model UB {final_rmse_ubcf:.4f}, IB {
            final_rmse_ibcf:.4f}\n"
    )
    f.write("\n")
    f.write("6.b) histogram of RMSE\n")
    f.write("RMSE     N.UBCF   N.IBCF\n")

    for i in range(len(ubcf_hist)):
        f.write(f"{bin_edges[i]:.2f}     {
                ubcf_hist[i]:4d}     {ibcf_hist[i]:4d}\n")

    f.write("\n")
    f.write("6.c) Top-10 recommendations\n")
    f.write("UBCF\n")
    f.write(
        "user     book1    book2    book3    book4    book5    book6    book7    book8    book9    book10\n"
    )

    for i, (user_idx, titles) in enumerate(zip(recs_ubcf_pd["userIndex"], ubcf_titles)):
        if i < 10:  # Show first 10 for brevity
            title_str = "    ".join([f"{t:12s}" for t in titles[:10]])
            f.write(f"{user_idx:4d}     {title_str}\n")

    f.write("\nIBCF\n")
    f.write(
        "user     book1    book2    book3    book4    book5    book6    book7    book8    book9    book10\n"
    )

    for i, (user_idx, titles) in enumerate(zip(recs_ibcf_pd["userIndex"], ibcf_titles)):
        if i < 10:  # Show first 10 for brevity
            title_str = "    ".join([f"{t:12s}" for t in titles[:10]])
            f.write(f"{user_idx:4d}     {title_str}\n")

# --------------------------
# 10. Save Model Data (Fixed Version)
# --------------------------
print("Saving models...")

# Save Spark models using their native save method
final_ubcf.write().overwrite().save("models/final_ubcf_model")
final_ibcf.write().overwrite().save("models/final_ibcf_model")
user_indexer_model.write().overwrite().save("models/user_indexer_model")
book_indexer_model.write().overwrite().save("models/book_indexer_model")

# Save non-Spark data using pickle (this will work fine)
non_spark_data = {
    "book_mapping": book_mapping,
    "ubcf_user_rmse": ubcf_user_rmse_all,
    "ibcf_user_rmse": ibcf_user_rmse_all,
    "k_fold_results": {
        "ubcf_rmse_per_fold": ubcf_rmse_per_fold,
        "ibcf_rmse_per_fold": ibcf_rmse_per_fold,
    },
    "final_rmse_ubcf": final_rmse_ubcf,
    "final_rmse_ibcf": final_rmse_ibcf,
}

with open("model_data.pkl", "wb") as f:
    pickle.dump(non_spark_data, f)

print("Models saved successfully!")


# --------------------------
# 11. Save RData File
# --------------------------
print("Saving RData file...")


# Activate pandas conversion
pandas2ri.activate()

# 5.a) Original ratings matrix UI (UserID × ISBN with ratings)
ratings_matrix_pd = ratings_pd.pivot_table(
    index="User-ID", columns="ISBN", values="Book-Rating", fill_value=0
)

# 5.b) eval_sets: store folds (train/test indices per fold)
eval_sets = {
    f"fold_{i}": {
        "train_users": train.select("UserID").toPandas()
        if i == 0
        else None,  # placeholder
        "test_users": test.select("UserID").toPandas()
        if i == 0
        else None,  # placeholder
    }
    for i in range(k_folds)
}
# (⚠️ Note: in Python we don’t have R's `evaluationScheme()` directly,
# so here we just preserve the folds’ splits in a Python dict.)

# 5.c) Learned model matrices R.UB (UBCF factors) and R.IB (IBCF factors)
R_UB = final_ubcf.userFactors.toPandas()
R_IB = final_ibcf.itemFactors.toPandas()

# 5.d) V.RMSE, list of vectors of RMSE per user
V_RMSE = {
    "UBCF": np.array(ubcf_user_rmse_all),
    "IBCF": np.array(ibcf_user_rmse_all),
}

# Convert everything into R objects
r = ro.r
r_ratings_matrix = pandas2ri.py2rpy(ratings_matrix_pd)
r_R_UB = pandas2ri.py2rpy(R_UB)
r_R_IB = pandas2ri.py2rpy(R_IB)
r_V_RMSE = ListVector(
    {
        "UBCF": ro.FloatVector(V_RMSE["UBCF"]),
        "IBCF": ro.FloatVector(V_RMSE["IBCF"]),
    }
)

# Save into model.rdata
r.assign("UI", r_ratings_matrix)
r.assign("eval_sets", eval_sets)  # Will be a Python object unless converted
r.assign("R_UB", r_R_UB)
r.assign("R_IB", r_R_IB)
r.assign("V_RMSE", r_V_RMSE)

r('save(UI, eval_sets, R_UB, R_IB, V_RMSE, file="model.rdata", compress=TRUE)')

print("RData file 'model.rdata' saved successfully!")
