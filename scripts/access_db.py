from scripts.load_sql import execute_query_pandas
import os


class AccessDatabase:
    def __init__(self):
        self.path = os.path.join('data', 'dataset.db')

    # Get all articles
    def get_all_articles(self):
        query = "SELECT * FROM Articles"
        data = execute_query_pandas(self.path, query)
        print(f"-> Found {len(data)} articles")
        return data

    @staticmethod
    def parse_df_to_dict(input_df):
        return input_df.to_dict(orient='records')


# For Testing purposes
# data_access = AccessDatabase()
# data_df = data_access.get_all_articles()
# data_rows = data_access.parse_df_to_dict(data_df)
