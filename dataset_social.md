# Reddit
[Dataset](https://drive.google.com/file/d/1GbGX6WK_MiguVU0WGBNRgyS_MugpyJqw/view?usp=sharing)
В этом датасете находится первичная выгрузка 3568 постов и 56831 комментариев из финансовых сабреддитов.
Так как Reddit не позволяет выгружать данные за произвольный период в прошлом, а лишь определённое число последних постов, датасет планируется периодически дополнять, чтобы
собрать набор данных за более репрезентативный период времени. Данные выгружались из сабреддитов:
* r/investing
* r/stocks
* r/stockmarkets
* r/wallstreetbets
* r/daytrading
* r/options
* r/algotrading
* r/valueinvesting

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
| 15 | doc_embedding         | Embeddings created from the text content |
