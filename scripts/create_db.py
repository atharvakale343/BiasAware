import os
import sys
sys.path.insert(0, os.getcwd())
from scripts.load_sql import LoadSQL
from webxplore import WebScraper
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from scripts.models import Article, create_table

class CreateDatabase:
    def __init__(self):
        db_path = os.path.join('data', 'dataset.db')
        # Connect to the database using SQLAlchemy
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker()
        Session.configure(bind=engine)
        self.session = Session()
        create_table(engine=engine)

    def populate_db(self, sample_size=1000):
        data_provider = LoadSQL()
        articles = data_provider.get_dataset(sample_size=sample_size)
        for i, record in enumerate(articles.to_dict(orient='records')):
            try:
                text = WebScraper.ScrapeWebsite(record['url'], timeout=5).text_content
                if not i%25:
                    print(i)
                article = Article(source=record['source'], text=text, label=int(record['label']))
                self.session.add(article)
            except Exception as e:
                print(f'Error Encountered: {e}')
                continue
        self.session.commit()

    def close_db_conn(self):
        self.session.close()


# For testing purposes
# db = CreateDatabase()
# db.populate_db(sample_size=10_000)
# db.close_db_conn()