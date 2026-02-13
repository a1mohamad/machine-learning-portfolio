import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from app.utils import safe_prepare

def prepare(df, train_dict):
    df[['release_month', 'release_day', 'release_year']] = df['release_date'].str.split('/', expand=True).replace(np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[(df['release_year'] <= 19) & (df['release_year'] < 100), 'release_year'] += 2000
    df.loc[(df['release_year'] > 19) & (df['release_year'] < 100), 'release_year'] += 1000
    releaseDate = pd.to_datetime(df['release_date'])
    df['release_daysofweek'] = releaseDate.dt.dayofweek
    df['release_quarter'] = releaseDate.dt.quarter

    rating_na = df.groupby(['release_year', 'original_language'])['rating'].mean().reset_index()
    df = df.merge(rating_na, how='left', on=["release_year", "original_language"], suffixes=('', '_mean'))
    df['rating'] = df['rating'].fillna(df['rating_mean'])
    df.drop(columns='rating_mean', inplace=True)

    totalvotes_na = df.groupby(['release_year', 'original_language'])['totalVotes'].mean().reset_index()
    df = df.merge(totalvotes_na, how='left', on=['release_year', 'original_language'], suffixes=('', '_mean'))
    df['totalVotes'] = df['totalVotes'].fillna(df['totalVotes_mean'])
    df.drop(columns='totalVotes_mean', inplace=True)

    df['weightedRating'] = (df['rating'] * df['totalVotes'] + 6.367 * 1000) / (df['totalVotes'] + 1000)
    df['originalBudget'] = df['budget']
    df['inflationBudget'] = df['budget'] + df['budget'] * 1.8 / 100 * (2018 - df['release_year'])
    df['budget'] = np.log1p(df['budget'])

    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if isinstance(i, dict) and i.get('gender', None) == 0]) if isinstance(x, list) else 0)
    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if isinstance(i, dict) and i.get('gender', None) == 1]) if isinstance(x, list) else 0)
    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if isinstance(i, dict) and i.get('gender', None) == 2]) if isinstance(x, list) else 0)
    df['collection_name'] = df['belongs_to_collection'].apply(lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else 0)
    le = LabelEncoder()
    le.fit(list(df['collection_name'].fillna('')))
    df['collection_name'] = le.transform(df['collection_name'].fillna('').astype(str))
    df['num_keyword'] = df['Keywords'].apply(lambda x: len(x) if isinstance(x, (list, dict)) else 0)
    df['num_cast'] = df['cast'].apply(lambda x: len(x) if isinstance(x, (list, dict)) else 0)

    df['popularity_mean_year'] = df['popularity'] / df.groupby('release_year')['popularity'].transform('mean')
    df['budget_runtime_ratio'] = df['budget'] / df['runtime']
    df['budget_populrity_ratio'] = df['budget'] / df['popularity']
    df['budget_year_ratio'] = df['budget'] / (df['release_year'] * df['release_year'])
    df['year_popularity_ratio1'] = df['release_year'] / df['popularity']
    df['year_populairty_ratio2'] = df['popularity'] / df['release_year']
    df['popularity_totalVotes_ratio'] = df['popularity'] / df['totalVotes']
    df['rating_popularity_ratio'] = df['rating'] / df['popularity']
    df['rating_totalVotes_ratio'] = df['rating'] / df['totalVotes']
    df['totalVotes_year_ratio'] = df['totalVotes'] / df['release_year']
    df['budget_rating_ratio'] = df['budget'] / df['rating']
    df['runtime_rating_ratio'] = df['runtime'] / df['rating']
    df['budget_totalVotes_ratio'] = df['budget'] / df['totalVotes']

    df['has_homepage'] = 1
    df.loc[pd.isnull(df['homepage']), 'has_homepage'] = 0
    df['isbelongs_to_collectionNA'] = 0
    df.loc[pd.isnull(df['belongs_to_collection']), 'isbelongs_to_collectionNA'] = 1
    df['isTaglineNA'] = 0
    df.loc[df['tagline'] == 0, 'isTaglineNA'] = 1
    df['isOriginalLanguageEN'] = 0
    df.loc[df['original_language'] == 'en', 'isOriginalLanguageEN'] = 1
    df['isTitleDifferent'] = 1
    df.loc[df['original_title'] == df['title'], 'isTitleDifferent'] = 0
    df['isReleased'] = 1
    df.loc[df['status'] != 'Released', 'isReleased'] = 0

    df['collection_id'] = df['belongs_to_collection'].apply(
        lambda x: np.nan if not isinstance(x, list) or len(x) == 0 else x[0]['id']
    )
    df['original_title_letter_count'] = df['original_title'].str.len()
    df['orignal_title_word_count'] = df['original_title'].str.split().str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    df['overview_word_count'] = df['overview'].str.split().str.len()
    df['tagline_word_count'] = df['tagline'].str.split().str.len()
    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['cast_count'] = df['cast'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['crew_count'] = df['crew'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['meanRuntimeByYear'] = df.groupby('release_day')['runtime'].transform('mean')
    df['meanPopularityByYear'] = df.groupby('release_year')['popularity'].transform('mean')
    df['meanBudgetByYear'] = df.groupby('release_year')['budget'].transform('mean')
    df['meantotalVotesByYear'] = df.groupby('release_year')['totalVotes'].transform('mean')
    df['medianBudgetByYear'] = df.groupby('release_year')['budget'].transform('median')

    for col in ['production_companies', 'production_countries', 'spoken_languages', 'genres']:
        df[col] = df[col].map(
            lambda x: sorted(list(set(
                [n if n in train_dict[col] else col + '_etc'
                for n in [d['name'] for d in x if isinstance(d, dict) and 'name' in d]]
            ))) if isinstance(x, list) else ''
        ).map(lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        df = pd.concat([df, temp], axis=1, sort=False)
    if 'genres_etc' in df.columns:
        df.drop('genres_etc', axis=1, inplace=True)
    df.drop(['title', 'cast', 'Keywords', 'spoken_languages',
            'production_companies', 'production_countries',
            'id', 'belongs_to_collection', 'revenue', 'overview',
            'original_language', 'poster_path', 'homepage',
            'runtime', 'genres', 'crew', 'imdb_id', 'release_date',
            'tagline', 'collection_id', 'original_title', 'status'], axis=1, inplace=True, errors='ignore')
    df.fillna(value=0.0, inplace=True)
    return df

# -- Load global stats and train_dict and features at module level

with open('models/global_stats.pkl', 'rb') as f:
    global_stats = pickle.load(f)

with open('models/train_dict.pkl', 'rb') as f:
    train_dict = pickle.load(f)

with open('models/features.pkl', 'rb') as f:
    features = pickle.load(f)


def prepare_single(df, train_dict, global_stats, features):
    # -- Date features
    df[['release_month', 'release_day', 'release_year']] = df['release_date'].str.split('/', expand=True).replace(np.nan, 0).astype(int)
    df['release_year'] = df['release_year']
    df.loc[(df['release_year'] <= 19) & (df['release_year'] < 100), 'release_year'] += 2000
    df.loc[(df['release_year'] > 19) & (df['release_year'] < 100), 'release_year'] += 1000
    releaseDate = pd.to_datetime(df['release_date'])
    df['release_daysofweek'] = releaseDate.dt.dayofweek
    df['release_quarter'] = releaseDate.dt.quarter

    # -- rating and totalVotes imputation
    def get_group_stat(stats_dict, year, lang, global_mean):
        return stats_dict.get((year, lang), global_mean)
    df['rating'] = df.apply(lambda row: row['rating'] if pd.notnull(row['rating'])
                            else get_group_stat(global_stats['rating_mean_by_year_lang'],
                                                row['release_year'], row['original_language'], global_stats["rating_mean"]), axis=1)
    df['totalVotes'] = df.apply(lambda row: row['totalVotes'] if pd.notnull(row['totalVotes'])
                                else get_group_stat(global_stats['totalVotes_mean_by_year_lang'],
                                                    row['release_year'], row['original_language'], global_stats["totalVotes_mean"]), axis=1)

    # -- weightedRating
    df['weightedRating'] = (df['rating'] * df['totalVotes'] + global_stats["weightedRating_mean"] * 1000) / (df['totalVotes'] + 1000)
    df['originalBudget'] = df['budget']
    df['inflationBudget'] = df['budget'] + df['budget'] * 1.8 / 100 * (2018 - df['release_year'])
    df['budget'] = np.log1p(df['budget'])

    # -- Crew gender counts
    df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if isinstance(i, dict) and i.get('gender', None) == 0]) if isinstance(x, list) else 0)
    df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if isinstance(i, dict) and i.get('gender', None) == 1]) if isinstance(x, list) else 0)
    df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if isinstance(i, dict) and i.get('gender', None) == 2]) if isinstance(x, list) else 0)

    # -- Collection name label encoding
    df['collection_name_raw'] = df['belongs_to_collection'].apply(lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else '0')
    le = LabelEncoder()
    le.classes_ = np.array(global_stats['collection_name_classes'])
    df['collection_name'] = le.transform(df['collection_name_raw'].fillna('').astype(str))
    df.drop(columns=['collection_name_raw'], inplace=True)

    # -- Keyword and cast counts
    df['num_keyword'] = df['Keywords'].apply(lambda x: len(x) if isinstance(x, (list, dict)) else global_stats['num_keyword_mean'])
    df['num_cast'] = df['cast'].apply(lambda x: len(x) if isinstance(x, (list, dict)) else global_stats['num_cast_mean'])

    # -- Popularity mean by year
    df['popularity_mean_year'] = df.apply(lambda row: row['popularity'] / global_stats['popularity_mean_by_year'].get(row['release_year'], global_stats["popularity_mean"]) if row['popularity'] else 0.0, axis=1)

    # -- Feature ratios (handle zero division)
    df['budget_runtime_ratio'] = df.apply(lambda row: row['budget'] / row['runtime'] if row['runtime'] else 0.0, axis=1)
    df['budget_populrity_ratio'] = df.apply(lambda row: row['budget'] / row['popularity'] if row['popularity'] else 0.0, axis=1)
    df['budget_year_ratio'] = df.apply(lambda row: row['budget'] / (row['release_year'] * row['release_year']) if row['release_year'] else 0.0, axis=1)
    df['year_popularity_ratio1'] = df.apply(lambda row: row['release_year'] / row['popularity'] if row['popularity'] else 0.0, axis=1)
    df['year_populairty_ratio2'] = df.apply(lambda row: row['popularity'] / row['release_year'] if row['release_year'] else 0.0, axis=1)
    df['popularity_totalVotes_ratio'] = df.apply(lambda row: row['popularity'] / row['totalVotes'] if row['totalVotes'] else 0.0, axis=1)
    df['rating_popularity_ratio'] = df.apply(lambda row: row['rating'] / row['popularity'] if row['popularity'] else 0.0, axis=1)
    df['rating_totalVotes_ratio'] = df.apply(lambda row: row['rating'] / row['totalVotes'] if row['totalVotes'] else 0.0, axis=1)
    df['totalVotes_year_ratio'] = df.apply(lambda row: row['totalVotes'] / row['release_year'] if row['release_year'] else 0.0, axis=1)
    df['budget_rating_ratio'] = df.apply(lambda row: row['budget'] / row['rating'] if row['rating'] else 0.0, axis=1)
    df['runtime_rating_ratio'] = df.apply(lambda row: row['runtime'] / row['rating'] if row['rating'] else 0.0, axis=1)
    df['budget_totalVotes_ratio'] = df.apply(lambda row: row['budget'] / row['totalVotes'] if row['totalVotes'] else 0.0, axis=1)

    # -- Boolean and indicator features
    df['has_homepage'] = df['homepage'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['isbelongs_to_collectionNA'] = df['belongs_to_collection'].apply(lambda x: 1 if pd.isnull(x) else 0)
    df['isTaglineNA'] = df['tagline'].apply(lambda x: 1 if x == 0 else 0)
    df['isOriginalLanguageEN'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
    df['isTitleDifferent'] = df.apply(lambda row: 0 if row['original_title'] == row['title'] else 1, axis=1)
    df['isReleased'] = df['status'].apply(lambda x: 1 if x == 'Released' else 0)

    # -- Collection id
    df['collection_id'] = df['belongs_to_collection'].apply(lambda x: np.nan if not isinstance(x, list) or len(x) == 0 else x[0]['id'])

    # -- Text/statistics features (fallback to mean if nan)
    df['original_title_letter_count'] = df['original_title'].str.len().fillna(global_stats['original_title_letter_count_mean'])
    df['orignal_title_word_count'] = df['original_title'].str.split().str.len().fillna(global_stats['original_title_word_count_mean'])
    df['title_word_count'] = df['title'].str.split().str.len().fillna(global_stats['title_word_count_mean'])
    df['overview_word_count'] = df['overview'].str.split().str.len().fillna(global_stats['overview_word_count_mean'])
    df['tagline_word_count'] = df['tagline'].str.split().str.len().fillna(global_stats['tagline_word_count_mean'])

    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(x) if isinstance(x, list) else global_stats['production_countries_count_mean'])
    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(x) if isinstance(x, list) else global_stats['production_companies_count_mean'])
    df['cast_count'] = df['cast'].apply(lambda x: len(x) if isinstance(x, list) else global_stats['cast_count_mean'])
    df['crew_count'] = df['crew'].apply(lambda x: len(x) if isinstance(x, list) else global_stats['crew_count_mean'])

    # -- Time-based group stats (fallback to mean)
    df['meanRuntimeByYear'] = df.apply(lambda row: global_stats['meanRuntimeByYear'].get(row['release_day'], global_stats['runtime_mean']), axis=1)
    df['meanPopularityByYear'] = df.apply(lambda row: global_stats['meanPopularityByYear'].get(row['release_year'], global_stats['popularity_mean']), axis=1)
    df['meanBudgetByYear'] = df.apply(lambda row: global_stats['meanBudgetByYear'].get(row['release_year'], global_stats['budget_mean']), axis=1)
    df['meantotalVotesByYear'] = df.apply(lambda row: global_stats['meantotalVotesByYear'].get(row['release_year'], global_stats['totalVotes_mean']), axis=1)
    df['medianBudgetByYear'] = df.apply(lambda row: global_stats['medianBudgetByYear'].get(row['release_year'], global_stats['budget_mean']), axis=1)

    # -- One-hot for categorical fields
    for col in ['production_companies', 'production_countries', 'spoken_languages', 'genres']:
        df[col] = df[col].map(
            lambda x: sorted(list(set(
                [n if n in train_dict[col] else col + '_etc'
                 for n in [d['name'] for d in x if isinstance(d, dict) and 'name' in d]]
            ))) if isinstance(x, list) else ''
        ).map(lambda x: ','.join(map(str, x)))
        temp = df[col].str.get_dummies(sep=',')
        df = pd.concat([df, temp], axis=1, sort=False)
    if 'genres_etc' in df.columns:
        df.drop('genres_etc', axis=1, inplace=True)

    # -- Drop all unused columns
    df.drop(['title', 'cast', 'Keywords', 'spoken_languages',
            'production_companies', 'production_countries',
            'id', 'belongs_to_collection', 'revenue', 'overview',
            'original_language', 'poster_path', 'homepage',
            'runtime', 'genres', 'crew', 'imdb_id', 'release_date',
            'tagline', 'collection_id', 'original_title', 'status'], axis=1, inplace=True, errors='ignore')

    df.fillna(value=0.0, inplace=True)

    # -- Ensure feature order and all required features present (add missing as 0)
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
    df = df[features]
    return df