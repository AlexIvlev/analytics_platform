from pandas import DataFrame

SOCIAL_EXPECTED_COLUMNS = ['id', 'title', 'url', 'subreddit', 'created_utc', 'parsed_utc', 'text', 'score',
                           'num_comments', 'type', 'parent_id', 'clean_text', 'processed_text',
                           'processed_text_length',
                           'sentiment_scores', 'entities', 'doc_embedding', 'tickers', 'ticker', 'created_price',
                           'price_1d']


def check_uploaded_data(df: DataFrame, dataset: str, check_size: bool = False) -> (bool, str):
    if df.shape[0] == 0:
        return False, "Загружен пустой датасет"
    if check_size and df.shape[0] < 100:
        return False, "Датасет содержит менее 100 наблюдений, загрузите датасет с большим количеством данных"
    if dataset == "Social 🧻":
        if df.shape[1] != len(SOCIAL_EXPECTED_COLUMNS):
            return False, f"Неверное количество столбцов." \
                          f" Ожидалось {len(SOCIAL_EXPECTED_COLUMNS)}, получено {df.shape[1]}"
        for col in SOCIAL_EXPECTED_COLUMNS:
            if col not in df.columns:
                return False, f"Отсутствует столбец {col}"
    return True, None
