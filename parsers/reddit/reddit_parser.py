import argparse
import logging
import os
import re
import sys
import time

import pandas as pd
import praw
import prawcore
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

all_subreddits = ['investing', 'stocks', 'stockmarkets', 'wallstreetbets', 'daytrading', 'options', 'algotrading',
                  'valueinvesting']


class RedditParser:
    columns = ["id", "title", "url", "subreddit", "created_utc", "parsed_utc", "text", "score", "num_comments", "type",
               "parent_id"]
    user_agent = "masters-degree.sentiment-analysis:v0.0.1 (by /u/Many-Necessary5937)"

    RATELIMIT_SECONDS = 600
    RATELIMIT_SLEEP_SECONDS = 60

    def __init__(self, subreddits: list, client_id: str, client_secret: str, limit: int = 1000) -> None:
        self.subreddits = subreddits
        self.limit = limit
        self.df = pd.DataFrame(columns=self.columns)
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=self.user_agent,
            check_for_async=False,
            ratelimit_seconds=self.RATELIMIT_SECONDS
        )

    def _get_rate_limits(self) -> dict:
        return self.reddit.auth.limits

    def parse(self) -> None:
        for subreddit in tqdm(self.subreddits, desc="Processing subreddits"):
            logging.info(f"Processing {subreddit}")
            entries = []
            try:
                for sub in tqdm(self.reddit.subreddit(subreddit).new(limit=self.limit), desc="Processing submissions"):
                    try:
                        entries.extend(self._process_submission(sub, subreddit))
                    except prawcore.exceptions.TooManyRequests:
                        logging.error(f"Rate limit exceeded while processing '{subreddit}'->'{sub.id}'")
                        logging.error("Sleeping for 1 minute and continuing from next submission")
                        time.sleep(self.RATELIMIT_SLEEP_SECONDS)
            except prawcore.exceptions.NotFound:
                logging.error(f"Subreddit '{subreddit}' not found")

            self.df = pd.concat([self.df, pd.DataFrame(entries, columns=self.df.columns)])

    def _process_submission(self, sub, subreddit) -> list:
        logging.info(self._get_rate_limits())

        entries = []
        created_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(sub.created_utc))
        parsed_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        data = {
            "id": sub.id,
            "title": sub.title,
            "url": sub.url,
            "subreddit": subreddit,
            "created_utc": created_utc,
            "parsed_utc": parsed_utc,
            "text": RedditParser.clean_comment(sub.selftext, remove_urls=True),
            "score": sub.score,
            "num_comments": sub.num_comments,
            "type": "submission",
            "parent_id": None
        }
        sub_row = pd.Series(data, index=self.df.columns)
        entries.append(sub_row)

        sub.comments.replace_more(limit=0)
        for comment in sub.comments:
            created_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(comment.created_utc))
            data = {
                "id": comment.id,
                "title": None,
                "url": None,
                "subreddit": subreddit,
                "created_utc": created_utc,
                "parsed_utc": parsed_utc,
                "text": RedditParser.clean_comment(comment.body, remove_urls=True),
                "score": comment.score,
                "num_comments": None,
                "type": "comment",
                "parent_id": sub.id
            }
            comm_row = pd.Series(data, index=self.df.columns)
            entries.append(comm_row)
        return entries

    def save_to_parquet(self, filename: str = 'reddit_parser') -> None:
        date_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.df.to_parquet(f'{filename}_{date_time}.parquet')

    @staticmethod
    def clean_comment(text: str, remove_urls: bool = False, remove_special_chars: bool = False,
                      remove_digits: bool = False, to_lower: bool = False) -> str:
        if remove_urls:
            text = re.sub(r'http\S+', '', text)
        if remove_special_chars:
            text = re.sub(r'[^\w\s]', '', text)
        # possibly breaks NER for some ticker symbols
        if remove_digits:
            text = re.sub(r'\d+', '', text)
        # possibly breaks NER for ticket symbols, so perform NER first
        if to_lower:
            text = text.lower()
        return text


def main():
    logging.info(f'Arguments passed: {sys.argv[1:]}')

    parser = argparse.ArgumentParser()

    parser.add_argument("--subreddits", nargs="+", help="List of subreddits to parse", required=False,
                        default=all_subreddits)
    parser.add_argument("--limit", type=int, help="Number of submissions to parse", required=False, default=1000)
    args = parser.parse_args()

    rp = RedditParser(
        client_id=os.getenv("client_id"),
        client_secret=os.getenv("client_secret"),
        subreddits=args.subreddits,
        limit=args.limit)

    rp.parse()
    rp.save_to_parquet()


if __name__ == "__main__":
    main()
