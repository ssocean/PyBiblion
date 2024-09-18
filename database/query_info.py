import os
import re
from tqdm import tqdm
from database.DBEntity import PaperMapping
from furnace.arxiv_paper import Arxiv_paper
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from furnace.arxiv_paper import Arxiv_paper
from furnace.google_scholar_paper import Google_paper
from tools.gpt_util import *
import time
import random

Base = declarative_base()

# Create a database engine
engine = create_engine('mysql+mysqlconnector://root:xxxx@localhost/ripami')

# Create database tables
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()


def statistics_database(session):
    rst = {}
    results = session.query(PaperMapping).all()
    years = []
    ref_counts = []
    citation_counts = []
    pubs = []
    authors_num = []
    for row in tqdm(results):
        if row.is_review == 1 and row.valid == 1:
            if row.publication_date.year is not None:
                years.append(int(row.publication_date.year))
            if row.s2_reference_count is not None and row.s2_reference_count >= 1:
                ref_counts.append(int(row.s2_reference_count))
            if row.s2_citation_count is not None:
                citation_counts.append(int(row.s2_citation_count))
            if row.s2_pub_info is not None:
                pubs.append(row.s2_pub_info)
            if row.authors_num is not None:
                authors_num.append(row.authors_num)
    rst['years'] = years
    rst['ref_counts'] = ref_counts
    rst['citation_counts'] = citation_counts
    rst['pubs'] = pubs
    rst['authors_num'] = authors_num
    return rst