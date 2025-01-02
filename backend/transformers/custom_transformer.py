from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(['id', 'title', 'url', 'created_utc', 'parsed_utc', 'text', 'parent_id', 'clean_text',
                    'processed_text', 'entities', 'tickers', 'price_1d', 'doc_embedding'], axis=1)
        X = X.fillna(0)
        return X
