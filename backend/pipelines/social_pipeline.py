from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(['id', 'title', 'url', 'created_utc', 'parsed_utc', 'text', 'parent_id', 'clean_text',
                    'processed_text', 'entities', 'tickers', 'price_1d', 'doc_embedding'], axis=1)
        X = X.fillna(0)
        return X


preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ['subreddit', 'type', 'ticker'])
    ],
    remainder='passthrough'
)
