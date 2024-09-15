import logging
import time
import urllib
from datetime import datetime
from urllib.error import URLError
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from arxiv import SortCriterion, SortOrder
import os
import re
from tqdm import tqdm
from database.DBEntity import PaperMapping
from furnace.Author import Author
from furnace.arxiv_paper import Arxiv_paper, get_arxiv_id_from_url
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from furnace.google_scholar_paper import Google_paper
from furnace.semantic_scholar_paper import S2paper
from tools.PDF_generator import render_pdf
from tools.gpt_util import *
import requests
import os.path
import arxiv  # 1.4.3
from tqdm import tqdm
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from urllib.error import URLError

Base = declarative_base()

engine = create_engine('mysql+mysqlconnector://root:xxxx@localhost/ripami')

Base.metadata.create_all(engine)


Session = sessionmaker(bind=engine)
session = Session()

import datetime
import os
from urllib.error import URLError

import requests
# from arxiv import arxiv
import arxiv
from bs4 import BeautifulSoup
from retry import retry
from tqdm import tqdm


def extract_arxiv_ids(url):
    headers = {
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Ensure the request was successful

    soup = BeautifulSoup(response.content, 'html.parser')
    dl = soup.find('dl', {'id': 'articles'})

    arxiv_ids = []
    for dt in dl.find_all('dt'):
        a_tag = dt.find('a', id=True)
        if a_tag:
            arxiv_ids.append(a_tag['id'])

    return arxiv_ids
def download_papers_webpage(filed='cs.CV',download_folder=None,max_downloads=200,num_threads=4):
    url = f'https://arxiv.org/list/{filed}/recent?skip=0&show={max_downloads}'
    aid = extract_arxiv_ids(url)
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    client = arxiv.Client()
    # 创建数据库引擎
    idlst = [i+'v1' for i in aid]
    search_by_id = arxiv.Search(id_list=idlst)
    arxiv_rst = []
    for result in tqdm(client.results(search_by_id)):
        arxiv_rst.append(result)

    print(f'Searching papers Done. {len(arxiv_rst)} papers found. Start Downloading.')
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(download_single_paper_LTS, result, download_folder) for result in arxiv_rst
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                arxiv_logger.info(result)
@retry(delay = 2, tries= 3)
def download_single_paper_LTS(result, download_folder):
    session = session_factory()
    try:
        arxiv_paper = Arxiv_paper(result, ref_type='entity')
        data = session.query(PaperMapping).filter(PaperMapping.arxiv_id == arxiv_paper.id).first()
        if data is None:
            if download_folder:
                if not ( os.path.exists(os.path.join(download_folder, result._get_default_filename()))):
                    arxiv_logger.info(f'{arxiv_paper.id} is not existed in the disk, try downloading it. ')
                    try:
                        result.download_pdf(dirpath=download_folder)
                    except URLError as e:
                        if os.path.exists(os.path.join(download_folder, result._get_default_filename())):
                            os.remove(os.path.join(download_folder, result._get_default_filename()))
                        arxiv_logger.warning(f'{result._get_default_filename()} download failed. Due to {e}. Retrying...')
                        raise URLError(f'{result._get_default_filename()} download failed. Due to {e}')
                    except KeyboardInterrupt:
                        arxiv_logger.info("Operation interrupted by user.")
                        if os.path.exists(os.path.join(download_folder, result._get_default_filename())):
                            os.remove(os.path.join(download_folder, result._get_default_filename()))
                        exit()
            doc = PaperMapping(arxiv_paper=arxiv_paper)

            doc.arxiv_authors = ', '.join([a.name for a in arxiv_paper.authors])
            doc_detail = PaperMapping_detail(
                idLiterature=doc.idLiterature, arxiv_paper=arxiv_paper,
            )
            doc.downloaded_pth = result._get_default_filename()
            session.add(doc)
            session.add(doc_detail)
            session.commit()
            os.rename(os.path.join(download_folder, result._get_default_filename()),os.path.join(download_folder,doc.arxiv_id+'.pdf'))
            doc.download_date = datetime.datetime.now()
            doc.valid = -1
            session.commit()
            return True

        else:
            arxiv_logger.info(f'Paper {data.idLiterature} is already exist in database')
            data.last_update_time = datetime.datetime.now()
            data.download_date = datetime.datetime.now()
            session.commit()
            return False
    except Exception as e:
        doc.valid = 0
        arxiv_logger.error(f"Error processing paper {result._get_default_filename()}: {str(e)}")
        raise e
    finally:
        session.close()
        return False