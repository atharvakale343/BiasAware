from sqlalchemy import Column, Integer, String, ForeignKey, Table, MetaData
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Article(Base):
    __tablename__ = "Articles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String)
    text = Column(String)
    label = Column(Integer)

def create_table(engine):
    if not engine.dialect.has_table(engine, 'Articles'):  # If table don't exist, Create.
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