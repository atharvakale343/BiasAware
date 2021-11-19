import sqlite3
import pandas as pd
import os
import warnings

# This script shows an example of how to use NELA-GT-2019 with sqlite3

# Execute a given SQL query on the database and return values
def execute_query(path, query):
    conn = sqlite3.connect(path)
    # execute query on database and retrieve them with fetchall
    results = conn.cursor().execute(query).fetchall()
    return results


# Execute query and load results into pandas dataframe
def execute_query_pandas(path, query):
    conn = sqlite3.connect(path)
    df = pd.read_sql_query(query, conn)
    return df


# Start here
class LoadSQL():
    # Make input command line arguments
    def __init__(self):
        self.path = os.path.join('data', 'nela-gt-2019.db')
        self.csv_path = os.path.join('data', 'labels.csv')

    # Get all sources
    def get_all_articles(self):
        query = "SELECT * FROM newsdata"
        data = execute_query_pandas(self.path, query)
        print(f"-> Found {len(data)} articles")
        return data

    # Query 1: select all articles from a specific source
    def articles_from_source(self, source="thenewyorktimes"):
        query = "SELECT * FROM newsdata WHERE source='%s'" % source
        data = execute_query_pandas(self.path, query)
        print("-> Found %d articles from %s" % (len(data), source))
        return data

    # Query 2: select articles from multiple sources
    def articles_from_sources(self, sources=['thenewyorktimes', 'cnn', 'foxnews']):
    # Note that we need to add extra quotes around each source's name
    # for the query to work properly e.g.: "'thenewyorktimes'"
        sources_str = ["'%s'" % s for s in sources]
        query = "SELECT * FROM newsdata WHERE source IN (%s)" % ",".join(sources_str)
        data = execute_query_pandas(self.path, query)
        print("-> Found %d articles from %s" % (len(data), ",".join(sources)))
        return data

    def get_labels(self):
        labels = dict()
        with open(self.csv_path) as fin:
            # Read out the header line from label file
            fin.readline()
            # Iterate over lines, taking the value from first column after name
            # i.e., aggregated label
            for line in fin:
                l = line.strip().split(",")
                source = l[0]
                if l[1] == "":  # NODATA for this entry, skip it
                    continue
                labels[source] = int(l[1])  # get value from last column (label)

            print("- Read labels for %d sources" % len(labels))
        return labels

    def get_dataset(self, sample_size=1000):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            articles = self.get_all_articles()
            labels = self.get_labels()
            articles['label'] = articles['source'].map(labels)
            articles.dropna(subset=['label'], inplace=True)
            return articles.groupby('label').apply(lambda x: x.sample(sample_size))


# For testing purposes
# data_provider = LoadSQL()
# data = data_provider.get_dataset()
