from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"
RESULTS_DIR = OUTPUT_ROOT / "final_results"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
REVIEW_COUNT_LABEL = "Review count"
YEARLY_TREND_NAME = "yearly review trend"
RATING_DISTRIBUTION_NAME = "rating distribution"


def find_input_file(folder_name, file_name):
    candidates = [
        OUTPUT_ROOT / "enriched_reviews" / folder_name / file_name,
        OUTPUT_ROOT / folder_name / file_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_table(folder_name, file_name, used_files, skipped_messages):
    path = find_input_file(folder_name, file_name)
    if path is None:
        skipped_messages.append(
            f"Required input unavailable: {folder_name}/{file_name}."
        )
        return None

    try:
        table = pd.read_csv(path)
    except Exception as error:
        skipped_messages.append(
            f"Could not read {path.relative_to(PROJECT_ROOT)}: {error}."
        )
        return None

    relative_path = str(path.relative_to(PROJECT_ROOT))
    if relative_path not in used_files:
        used_files.append(relative_path)
    return table


def find_column(table, *names):
    available = {str(column).strip().lower(): column for column in table.columns}
    for name in names:
        match = available.get(name.strip().lower())
        if match is not None:
            return match
    return None


def numeric_values(table, column):
    return pd.to_numeric(table[column], errors="coerce")


def unused_path(directory, file_name):
    path = directory / file_name
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    version = 2
    while True:
        candidate = directory / f"{stem}_{version}{suffix}"
        if not candidate.exists():
            return candidate
        version += 1


def save_figure(figure, file_name, visualizations):
    path = unused_path(VISUALIZATIONS_DIR, file_name)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    visualizations.append(str(path.relative_to(PROJECT_ROOT)))
    return path


def skip_analysis(name, reason, skipped_messages):
    message = f"Skipped {name}: {reason}."
    skipped_messages.append(message)
    print(message)


def analyze_products(used_files, skipped_messages, findings, visualizations):
    candidates = [
        ("product_performance.csv", ("asin", "product_id", "product")),
        ("product_popularity.csv", ("asin", "product_id", "product")),
        ("sql_top_products.csv", ("product_id", "asin", "product")),
    ]
    selected = None
    for file_name, product_names in candidates:
        table = load_table("product_analysis", file_name, used_files, skipped_messages)
        if table is None:
            continue
        product_column = find_column(table, *product_names)
        count_column = find_column(table, "review_count", "count")
        if product_column is not None and count_column is not None:
            selected = (table, product_column, count_column)
            break

    if selected is None:
        skip_analysis(
            "top product analysis",
            "no product table with product identifiers and review counts was available",
            skipped_messages,
        )
        return

    table, product_column, count_column = selected
    table = table.copy()
    table[count_column] = numeric_values(table, count_column)
    top_products = table.dropna(subset=[count_column]).nlargest(10, count_column)
    if top_products.empty:
        skip_analysis("top product chart", "the review counts were not numeric", skipped_messages)
        return

    top_products = top_products.sort_values(count_column)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(top_products[product_column].astype(str), top_products[count_column], color="#277da1")
    axis.set_title("Top Products by Review Count")
    axis.set_xlabel(REVIEW_COUNT_LABEL)
    axis.set_ylabel("Product ID")
    axis.grid(axis="x", alpha=0.25)
    save_figure(figure, "01_top_products.png", visualizations)

    findings.append(
        "Top products by review count: "
        + ", ".join(
            f"{row[product_column]} ({int(row[count_column]):,})"
            for _, row in top_products.sort_values(count_column, ascending=False).head(3).iterrows()
        )
        + "."
    )


def analyze_verified_purchases(used_files, skipped_messages, findings, visualizations):
    table = load_table(
        "review_analysis", "verified_vs_nonverified.csv", used_files, skipped_messages
    )
    if table is None:
        skip_analysis(
            "verified versus non-verified analysis",
            "the comparison CSV was unavailable",
            skipped_messages,
        )
        return

    type_column = find_column(table, "customer_type", "purchase_type", "verified_purchase")
    count_column = find_column(table, "review_count", "count")
    rating_column = find_column(table, "average_rating", "avg_rating", "rating")
    if type_column is None or count_column is None:
        skip_analysis(
            "verified versus non-verified analysis",
            "the CSV lacks a purchase-type or review-count column",
            skipped_messages,
        )
        return

    table = table.copy()
    table[count_column] = numeric_values(table, count_column)
    if rating_column is not None:
        table[rating_column] = numeric_values(table, rating_column)

    if rating_column is not None:
        figure, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[1].bar(table[type_column].astype(str), table[rating_column], color="#f9844a")
        axes[1].set_title("Average Rating")
        axes[1].set_ylabel("Average rating")
        axes[1].set_ylim(0, 5)
        axes[1].tick_params(axis="x", rotation=15)
        rating_line = "Average ratings: " + ", ".join(
            f"{row[type_column]} {row[rating_column]:.2f}"
            for _, row in table.dropna(subset=[rating_column]).iterrows()
        ) + "."
    else:
        figure, axes = plt.subplots(figsize=(8, 5))
        axes = [axes]
        rating_line = "Average-rating comparison skipped: no rating column was available."

    axes[0].bar(table[type_column].astype(str), table[count_column], color="#43aa8b")
    axes[0].set_title("Review Counts")
    axes[0].set_ylabel(REVIEW_COUNT_LABEL)
    axes[0].tick_params(axis="x", rotation=15)
    if rating_column is not None:
        axes[0].ticklabel_format(axis="y", style="plain")
    figure.suptitle("Verified and Non-Verified Purchases")
    figure.tight_layout()
    save_figure(figure, "02_verified_vs_nonverified.png", visualizations)

    count_line = "Review counts: " + ", ".join(
        f"{row[type_column]} {int(row[count_column]):,}"
        for _, row in table.dropna(subset=[count_column]).iterrows()
    ) + "."
    findings.extend([count_line, rating_line])


def analyze_yearly_trends(used_files, skipped_messages, findings, visualizations):
    table = load_table(
        "review_analysis", "review_trends_year.csv", used_files, skipped_messages
    )
    if table is None:
        skip_analysis(YEARLY_TREND_NAME, "the yearly trend CSV was unavailable", skipped_messages)
        return

    year_column = find_column(table, "year")
    count_column = find_column(table, "review_count", "count")
    if year_column is None or count_column is None:
        skip_analysis(
            YEARLY_TREND_NAME,
            "the CSV lacks a year or review-count column",
            skipped_messages,
        )
        return

    table = table.copy()
    table[year_column] = numeric_values(table, year_column)
    table[count_column] = numeric_values(table, count_column)
    yearly = table.dropna(subset=[year_column, count_column]).sort_values(year_column)
    if yearly.empty:
        skip_analysis(YEARLY_TREND_NAME, "the year/count values were not numeric", skipped_messages)
        return

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(yearly[year_column], yearly[count_column], color="#277da1", marker="o", linewidth=1.8)
    axis.set_title("Reviews Over Time")
    axis.set_xlabel("Year")
    axis.set_ylabel("Review count")
    axis.grid(alpha=0.25)
    save_figure(figure, "03_reviews_over_time.png", visualizations)

    peak = yearly.loc[yearly[count_column].idxmax()]
    findings.append(
        f"Yearly trend covers {int(yearly[year_column].min())} to "
        f"{int(yearly[year_column].max())}; the highest count was "
        f"{int(peak[count_column]):,} in {int(peak[year_column])}."
    )


def prepare_length_table(table, category_column, helpful_column, rating_column):
    table = table.copy()
    table[helpful_column] = numeric_values(table, helpful_column)
    if rating_column is not None:
        table[rating_column] = numeric_values(table, rating_column)

    category_order = [
        "Very Short (<50)",
        "Short (50-199)",
        "Medium (200-499)",
        "Long (500-999)",
        "Very Long (1000+)",
    ]
    category_positions = {category: index for index, category in enumerate(category_order)}
    table["_category_order"] = table[category_column].astype(str).map(category_positions)
    return table.sort_values("_category_order", na_position="last")


def create_review_length_chart(table, category_column, helpful_column, rating_column, visualizations):
    if rating_column is not None:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[1].bar(table[category_column], table[rating_column], color="#f9c74f")
        axes[1].set_title("Average Rating")
        axes[1].set_ylabel("Average rating")
        axes[1].set_ylim(0, 5)
        axes[1].tick_params(axis="x", rotation=25)
    else:
        figure, helpful_axis = plt.subplots(figsize=(9, 5))
        axes = [helpful_axis]

    axes[0].bar(table[category_column], table[helpful_column], color="#90be6d")
    axes[0].set_title("Average Helpful Votes")
    axes[0].set_ylabel("Average helpful votes")
    axes[0].tick_params(axis="x", rotation=25)
    figure.suptitle("Review Length, Helpfulness, and Rating")
    figure.tight_layout()
    save_figure(figure, "04_review_length.png", visualizations)


def summarize_review_length(table, category_column, helpful_column, findings):
    valid_rows = table.dropna(subset=[helpful_column])
    if valid_rows.empty:
        return
    best = valid_rows.loc[valid_rows[helpful_column].idxmax()]
    findings.append(
        f"Among review-length categories, {best[category_column]} had the "
        f"highest average helpful votes ({best[helpful_column]:.2f})."
    )
    findings.append("Review counts by length and category ratings are shown in the chart.")


def analyze_overall_review_length(used_files, skipped_messages, findings):
    overall = load_table(
        "review_analysis", "review_length_analysis.csv", used_files, skipped_messages
    )
    if overall is None:
        return

    correlation_column = find_column(
        overall, "length_helpfulness_correlation", "review_length_helpfulness_correlation"
    )
    average_length_column = find_column(overall, "average_review_length", "avg_review_length")
    if correlation_column is not None:
        correlation = pd.to_numeric(overall[correlation_column], errors="coerce").dropna()
        if not correlation.empty:
            findings.append(
                f"Reported review-length/helpfulness correlation: {correlation.iloc[0]:.4f}."
            )
    if average_length_column is not None:
        average_length = pd.to_numeric(overall[average_length_column], errors="coerce").dropna()
        if not average_length.empty:
            findings.append(f"Average review length: {average_length.iloc[0]:.2f} characters.")


def analyze_review_length(used_files, skipped_messages, findings, visualizations):
    table = load_table(
        "review_analysis", "helpfulness_by_review_length.csv", used_files, skipped_messages
    )
    if table is None:
        skip_analysis(
            "review length and helpfulness analysis",
            "the review-length category CSV was unavailable",
            skipped_messages,
        )
    else:
        category_column = find_column(table, "review_length_category", "length_category")
        helpful_column = find_column(table, "average_helpful_votes", "avg_helpful_votes")
        rating_column = find_column(table, "average_rating", "avg_rating")
        if category_column is None or helpful_column is None:
            skip_analysis(
                "review length and helpfulness chart",
                "the CSV lacks review-length categories or average helpful votes",
                skipped_messages,
            )
        else:
            table = prepare_length_table(table, category_column, helpful_column, rating_column)
            create_review_length_chart(
                table, category_column, helpful_column, rating_column, visualizations
            )
            summarize_review_length(table, category_column, helpful_column, findings)

    analyze_overall_review_length(used_files, skipped_messages, findings)


def analyze_rating_distribution(used_files, skipped_messages, findings, visualizations):
    table = load_table(
        "review_analysis", "most_helpful_reviews.csv", used_files, skipped_messages
    )
    if table is None:
        skip_analysis(
            RATING_DISTRIBUTION_NAME,
            "the most-helpful reviews CSV was unavailable",
            skipped_messages,
        )
        return

    rating_column = find_column(table, "rating", "overall")
    if rating_column is None:
        skip_analysis(
            "rating distribution",
            "no rating column was present in the available review output",
            skipped_messages,
        )
        return

    ratings = numeric_values(table, rating_column).dropna()
    if ratings.empty:
        skip_analysis(RATING_DISTRIBUTION_NAME, "the rating values were unavailable", skipped_messages)
        return

    counts = ratings.value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(counts.index.astype(str), counts.values, color="#f9844a")
    axis.set_title("Ratings in the Most-Helpful Reviews Output")
    axis.set_xlabel("Rating")
    axis.set_ylabel("Number of selected reviews")
    axis.grid(axis="y", alpha=0.25)
    save_figure(figure, "05_rating_distribution.png", visualizations)

    findings.append(
        f"Rating distribution uses {len(ratings):,} rows from most_helpful_reviews.csv only; "
        "it is not a dataset-wide rating distribution. "
        f"Most common rating in this subset: {counts.idxmax()} ({int(counts.max()):,} rows)."
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    used_files = []
    skipped_messages = []
    findings = []
    visualizations = []

    analyze_products(used_files, skipped_messages, findings, visualizations)
    analyze_verified_purchases(used_files, skipped_messages, findings, visualizations)
    analyze_yearly_trends(used_files, skipped_messages, findings, visualizations)
    analyze_review_length(used_files, skipped_messages, findings, visualizations)
    analyze_rating_distribution(used_files, skipped_messages, findings, visualizations)

    summary_path = unused_path(RESULTS_DIR, "person4_summary.txt")
    summary_lines = [
        "Person 4 Advanced Analysis Summary",
        "",
        "Analysis performed:",
        "- Product popularity, verified-purchase comparisons, yearly review trends, review length/helpfulness, and available rating information.",
        "",
        "Person 3 CSV files used:",
    ]
    summary_lines.extend(f"- {path}" for path in used_files or ["No input CSV files were read."])
    summary_lines.extend(["", "Important findings:"])
    summary_lines.extend(f"- {finding}" for finding in findings or ["No findings could be calculated."])
    summary_lines.extend(["", "Visualizations created:"])
    summary_lines.extend(f"- {path}" for path in visualizations or ["No visualizations could be created."])
    summary_lines.extend(["", "Skipped analyses / data limitations:"])
    summary_lines.extend(
        f"- {message}"
        for message in skipped_messages or ["None; all performed analyses had the required data."]
    )
    summary_lines.extend(
        [
            "",
            "Input lookup prefers output/enriched_reviews/ and falls back to existing output analysis folders.",
            f"This summary: {summary_path.relative_to(PROJECT_ROOT)}",
        ]
    )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Person 4 analysis complete. Summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Visualizations saved: {len(visualizations)}")


if __name__ == "__main__":
    main()