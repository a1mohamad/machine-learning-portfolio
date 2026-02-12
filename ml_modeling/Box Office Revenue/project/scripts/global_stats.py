import pickle
import numpy as np
import pandas as pd

# ---- LOAD YOUR TRAINING DATA ----

train = pd.read_csv("./data/ultimate_train.csv")  # Adjust this as needed


# ---- GLOBAL STATISTICS FOR IMPUTATION AND SINGLE ROW PREP ----
global_stats = {}

# Means and medians for basic numeric fields
train[['release_month', 'release_day', 'release_year']] = train['release_date'].str.split('/', expand=True).replace(np.nan, 0).astype(int)
train['release_year'] = train['release_year']
train.loc[(train['release_year'] <= 19) & (train['release_year'] < 100), 'release_year'] += 2000
train.loc[(train['release_year'] > 19) & (train['release_year'] < 100), 'release_year'] += 1000
global_stats["rating_mean"] = train["rating"].mean()
global_stats["totalVotes_mean"] = train["totalVotes"].mean()
global_stats["runtime_mean"] = train["runtime"].mean()
global_stats["budget_mean"] = train["budget"].mean()
global_stats["popularity_mean"] = train["popularity"].mean()
global_stats["release_year_mean"] = train["release_year"].mean()
global_stats["release_day_mean"] = train["release_day"].mean()

# Means by (release_year, original_language) for rating and totalVotes
global_stats["rating_mean_by_year_lang"] = (
    train.groupby(["release_year", "original_language"])["rating"].mean().to_dict()
)
global_stats["totalVotes_mean_by_year_lang"] = (
    train.groupby(["release_year", "original_language"])["totalVotes"].mean().to_dict()
)

# Means by release_year for popularity, budget, totalVotes
global_stats["popularity_mean_by_year"] = train.groupby("release_year")["popularity"].mean().to_dict()
global_stats["budget_mean_by_year"] = train.groupby("release_year")["budget"].mean().to_dict()
global_stats["totalVotes_mean_by_year"] = train.groupby("release_year")["totalVotes"].mean().to_dict()

# Means by release_day for runtime
global_stats["runtime_mean_by_day"] = train.groupby("release_day")["runtime"].mean().to_dict()

# Medians by release_year for budget
global_stats["budget_median_by_year"] = train.groupby("release_year")["budget"].median().to_dict()

# WeightedRating constant (from your formula)
global_stats["weightedRating_mean"] = 6.367

# Default counts for fields based on your prepare logic
global_stats["num_keyword_mean"] = train["Keywords"].apply(lambda x: len(x) if isinstance(x, (list, dict)) else 0).mean()
global_stats["num_cast_mean"] = train["cast"].apply(lambda x: len(x) if isinstance(x, (list, dict)) else 0).mean()
global_stats["production_countries_count_mean"] = train["production_countries"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
global_stats["production_companies_count_mean"] = train["production_companies"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
global_stats["cast_count_mean"] = train["cast"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
global_stats["crew_count_mean"] = train["crew"].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()

# Text field length means
global_stats["original_title_letter_count_mean"] = train["original_title"].str.len().mean()
global_stats["original_title_word_count_mean"] = train["original_title"].str.split().str.len().mean()
global_stats["title_word_count_mean"] = train["title"].str.split().str.len().mean()
global_stats["overview_word_count_mean"] = train["overview"].str.split().str.len().mean()
global_stats["tagline_word_count_mean"] = train["tagline"].str.split().str.len().mean()

# Grouped means/medians for feature engineering in your pipeline
global_stats["meanRuntimeByYear"] = train.groupby("release_day")["runtime"].mean().to_dict()
global_stats["meanPopularityByYear"] = train.groupby("release_year")["popularity"].mean().to_dict()
global_stats["meanBudgetByYear"] = train.groupby("release_year")["budget"].mean().to_dict()
global_stats["meantotalVotesByYear"] = train.groupby("release_year")["totalVotes"].mean().to_dict()
global_stats["medianBudgetByYear"] = train.groupby("release_year")["budget"].median().to_dict()

# For collection_name LabelEncoder: fit on all names (for single-row encoding)
collection_names = train["belongs_to_collection"].apply(
    lambda x: x[0]["name"] if isinstance(x, list) and len(x) > 0 else "0"
)
global_stats["collection_name_classes"] = collection_names.fillna("").astype(str).unique().tolist()

# Save the stats with the required name
with open("models/global_stats.pkl", "wb") as f:
    pickle.dump(global_stats, f)

print("Global stats saved to models/global_stats.pkl")