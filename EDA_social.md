# EDA на датасете с постами и комментариями из Reddit

Ноутбук с EDA расположен по [ссылке](https://github.com/AlexIvlev/analytics_platform/blob/feature/reddit_parser/notebooks/eda/YP_2024_Reddit_dataset_EDA.ipynb).
### Структура данных

| #  | Column                | Description                    |
|----|------------------------|--------------------------------|
| 0  | id                    | Unique identifier for each entry |
| 1  | title                 | Title of the post or comment   |
| 2  | url                   | URL link to the post or comment |
| 3  | subreddit             | Subreddit where the post was made |
| 4  | created_utc           | Original creation timestamp in UTC |
| 5  | parsed_utc            | Parsed timestamp in UTC format |
| 6  | text                  | Content text of the post or comment |
| 7  | score                 | Score or rating of the post or comment |
| 8  | num_comments          | Number of comments on the post |
| 9  | type                  | Type of entry (e.g., post or comment) |
| 10 | parent_id             | Identifier of the parent post or comment |
| 11 | processed_text        | Processed version of the text content |
| 12 | processed_text_length | Length of the processed text |
| 13 | sentiment_scores      | Sentiment score of the text content |
| 14 | entities              | Extracted entities from the text content |

Исходные данные представляют собой тексты постов и комментариев пользователей. Предобработка помогла очистить данные от английских стоп-слов, пустых и бессмысленных с точки зрения нашей задачи текстов.
Данные были обогащены оценкой сентимента текста, выделенными организациями, о которых говорится в тексте.

Статистический анализ слов, биграмм, триграмм после очистки выявил, что в топах везде присутствуют связанные с финансовым рынком термины, что говорит о правильности выбора источников данных для анализа.



### Потенциальные проблемы
* В датасете присутствуют комментарии пользователей со всего света, поэтому попадаются самые разные языки, в том числе использующие латиницу. Так как основные модели (удаление стоп-слов, выделение сентимента) обычно ориентируются на какой-то конкретный язык, и планируется использование моделей только для английского языка, это может вызвать понижение точности.
* Такие задачи как уменьшение размерности с помощью t-SNE, генерация эмбеддингов с помощью Doc2Vec крайне ресурсоёмки и выполняются долго на доступных нам бесплатных ресурсах. Пока это блокер для дальнейшего увеличения датасета.
