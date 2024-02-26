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

def add_info_to_database(dir:str, session):
    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)
            if match:
                arxiv_id = match.group()
                cur_paper = Arxiv_paper(arxiv_id, ref_type='id')
                _ = cur_paper.entity

                doc = PaperMapping(cur_paper)

                # ... Set other properties

                # Add the object to the session
                session.add(doc)

                # Commit changes to persist the object to the database
                session.commit()
            else:
                print("No arXiv ID found.")

    # Close the session
    session.close()

def update_keyword(dir:str, session):
    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)

            title_parts = filename.split('.')
            if match:
                arxiv_id = '.'.join(title_parts[:2])
                id = 'http://arxiv.org/abs/'+arxiv_id

                data = session.query(PaperMapping).filter(PaperMapping.id == id).first()
                if data and data.gpt_keywords is None:
                    time.sleep(random.uniform(5.5, 8))
                    try:
                        kwd = get_chatgpt_keyword(data.title, data.abstract.replace('\n',''))
                    except Exception as e:
                        print(e)
                        print(data.title)
                        print(data.abstract)
                        return
                    data.gpt_keywords = ','.join(kwd)

                    try:
                        if data.citation_count is None:
                            g_paper = Google_paper(data.title)

                            if g_paper.citation_count >= 0:
                                data.citation_count = g_paper.citation_count

                            if g_paper.publication_source != 'NA':
                                data.publication_source = g_paper.publication_source

                    except KeyError as e:
                        print(e)
                        return
                    
                    session.commit()
                else:
                    pass
            session.close()

    session.close()

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