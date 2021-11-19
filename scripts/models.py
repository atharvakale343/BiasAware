from sqlalchemy import Column, Integer, String, Table, MetaData, inspect
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Article(Base):
    __tablename__ = "Articles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String)
    text = Column(String)
    label = Column(Integer)


def create_table(engine):
    if not inspect(engine).has_table('Articles'):
        metadata = MetaData(engine)
        # Create a table with the appropriate Columns
        Table('Articles', metadata,
              Column('id', Integer, primary_key=True, autoincrement=True),
              Column('source', String),
              Column('text', String),
              Column('label', Integer),
              )
        # Implement the creation
        metadata.create_all()
