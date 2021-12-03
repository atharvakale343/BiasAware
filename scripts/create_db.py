import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())
from scripts.load_sql import LoadSQL
from webxplore import WebScraper
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from scripts.models import Article, create_table
from asgiref.sync import sync_to_async


class CreateDatabase:
    def __init__(self):
        db_path = os.path.join('data', 'dataset.db')
        # Connect to the database using SQLAlchemy
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker()
        Session.configure(bind=engine)
        self.session = Session()
        create_table(engine=engine)

    async def populate_db(self, sample_size=1000):
        data_provider = LoadSQL()
        articles = data_provider.get_dataset(sample_size=sample_size)
        list_tasks = []
        for record in articles.to_dict(orient='records'):
            list_tasks.append(self.return_article_content(record))

        list_output = await asyncio.gather(*list_tasks)

        for record, output in list_output:
            article = Article(source=record['source'], text=output, label=int(record['label']))
            self.session.add(article)

        self.session.commit()

    async def return_article_content(self, record):
        try:
            return_t = await sync_to_async(WebScraper.ScrapeWebsite)(record['url'], timeout=5)
            return_t = return_t.text_content
        except Exception as e:
            print(f'Error Encountered: {e}')
            return ''
        return record, return_t

    def close_db_conn(self):
        self.session.close()


# For testing purposes
# db = CreateDatabase()
# asyncio.run(db.populate_db(sample_size=10000))
# db.close_db_conn()
