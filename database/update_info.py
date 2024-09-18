from datetime import datetime
import os
import re
import shelve

import requests
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

from CACHE.CACHE_Config import generate_cache_file_name
from database.DBEntity import PaperMapping, CoP
from furnace.Author import Author
from furnace.arxiv_paper import Arxiv_paper

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from furnace.google_scholar_paper import Google_paper
from furnace.semantic_scholar_paper import S2paper
from tools.gpt_util import *
from cfg.safesession import session, engine, session_factory


def update_keyword(dir: str, session):
    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)

            title_parts = filename.split('.')
            if match:
                arxiv_id = '.'.join(title_parts[:2])
                id = 'http://arxiv.org/abs/' + arxiv_id

                data = session.query(PaperMapping).filter(PaperMapping.id == id).first()
                if data and data.gpt_keywords is None:
                    time.sleep(random.uniform(5.5, 8))
                    try:
                        kwd = get_chatgpt_keyword(data.title, data.abstract.replace('\n', ''))
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
def add_info_to_database(dir: str, session):
    '''
    从arxiv下载后论文的第一步
    :param dir:
    :param session:
    :return:
    '''
    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)
            title_parts = filename.split('.')
            if match:
                # arxiv_id = match.group()
                arxiv_id = '.'.join(title_parts[:2])
                cur_paper = Arxiv_paper(arxiv_id, ref_type='id')

                id = 'http://arxiv.org/abs/' + arxiv_id
                data = session.query(PaperMapping).filter(PaperMapping.id == id).first()
                # print(data.idLiterature)
                if data is None:
                    print(arxiv_id)
                    _ = cur_paper.entity

                    doc = PaperMapping(arxiv_paper=cur_paper)

                    session.add(doc)

                    session.commit()
                else:
                    print('done')

            else:
                print("No arXiv ID found.")

    session.close()


import time
import random


def update_keyword(dir: str, session):
    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            # print(filename)
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)

            title_parts = filename.split('.')
            if match:
                arxiv_id = '.'.join(title_parts[:2])
                id = 'http://arxiv.org/abs/' + arxiv_id

                data = session.query(PaperMapping).filter(PaperMapping.id == id).first()
                # print(data.idLiterature)
                if data and data.gpt_keywords is None:
                    time.sleep(random.uniform(5.5, 8))
                    try:
                        kwd = get_chatgpt_keyword(data.title, data.abstract.replace('\n', ''))
                    except Exception as e:
                        print(e)
                        print(data.title)
                        print(data.abstract)
                        return
   
                    # kwd=''
                    data.gpt_keywords = ','.join(kwd)

                    # print(','.join(kwd))
                    try:

                        if data.citation_count is None:
                            g_paper = Google_paper(data.title)

                            if g_paper.citation_count >= 0:
                                data.citation_count = g_paper.citation_count
                                # print(data.citation_count)
                            if g_paper.publication_source != 'NA':
                                data.publication_source = g_paper.publication_source
                                # print(data.publication_source)

                    except KeyError as E:
                        print(E)
                        return
                    #
                    session.commit()

                else:
                    pass
    session.close()


import json


def get_keywords():
    results = session.query(PaperMapping).all()
    rst = []
    for row in tqdm(results):
        rst += row.gpt_keywords.split(',')
    session.close()
    return rst


from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
import time
from datetime import datetime


# def update_s2(session, mandatory_update=False):
#     # Get list of IDs to process
#     results = session.query(PaperMapping).all()
#     ids_to_process = [
#         row.id for row in results
#         if (row.s2_id is None or mandatory_update) and row.is_review == 1
#     ]
#
#     # Create a session factory bound to the same engine
#     SessionFactory = sessionmaker(bind=session.get_bind())
#
#     # Define the worker function
#     def process_row(row_id):
#         thread_session = SessionFactory()
#         try:
#             row = thread_session.query(PaperMapping).get(row_id)
#             if not row:
#                 return  # Skip if row is not found
#
#             s2_paper = S2paper(row.title, filled_authors=False, force_return=True)
#             if s2_paper.s2id is not None:
#                 if s2_paper.title != row.title:
#                     if s2_paper.authors and s2_paper.authors[0].name in row.authors:
#                         print(
#                             f'The same paper with different titles detected: {row.title} \n {s2_paper.title}'
#                         )
#                     else:
#                         return  # Skip to next if titles don't match
#
#                 time.sleep(0.5)
#                 # Update the row with data from s2_paper
#                 row.s2_id = s2_paper.s2id
#                 row.s2_publication_date = s2_paper.publication_date
#                 row.s2_tldr = s2_paper.tldr
#                 row.s2_DOI = s2_paper.DOI
#                 row.s2_pub_info = s2_paper.publication_source
#                 row.s2_pub_url = s2_paper.pub_url['url'][:255] if s2_paper.pub_url else None
#                 row.s2_citation_count = s2_paper.citation_count
#                 row.s2_reference_count = s2_paper.reference_count
#                 row.s2_field = s2_paper.field
#                 row.s2_influential_citation_count = s2_paper.influential_citation_count
#                 row.valid = 1
#                 row.last_update_time = datetime.now()
#                 row.authors_num = len(s2_paper.authors) if s2_paper.authors else None
#             else:
#                 row.valid = 0
#
#             thread_session.commit()
#         except Exception as e:
#             thread_session.rollback()
#             print(f"Error processing row {row_id}: {e}")
#         finally:
#             thread_session.close()
#
#     # Use ThreadPoolExecutor for parallel processing
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         list(tqdm(executor.map(process_row, ids_to_process), total=len(ids_to_process)))
#
#     session.close()
def update_s2(session, mandatory_update=False):
    results = session.query(PaperMapping).filter(PaperMapping.s2_citation_count.is_(None)).all()
    for row in tqdm(results):
        if (row.s2_id is None or mandatory_update) and row.is_review==1:
            # if row.authors_num is not None:
            #     continue
            s2_paper = S2paper(row.title, filled_authors=False, force_return=True)
            if s2_paper.s2id is not None:

                if s2_paper.title != row.title:
                    if s2_paper.authors[0].name in row.authors:
                        print(f'The same paper with different titles detected: {row.title} \n {s2_paper.title}')
                    else:
                        continue
                time.sleep(0.5)
                row.s2_id = s2_paper.s2id
                row.s2_publication_date = s2_paper.publication_date
                row.s2_tldr = s2_paper.tldr
                row.s2_DOI = s2_paper.DOI
                row.s2_pub_info = s2_paper.publication_source
                row.s2_pub_url = s2_paper.pub_url['url'][:255] if s2_paper.pub_url else None
                row.s2_citation_count = s2_paper.citation_count
                row.s2_reference_count = s2_paper.reference_count
                row.s2_field = s2_paper.field
                row.s2_influential_citation_count = s2_paper.influential_citation_count
                row.valid = 1
                row.last_update_time = datetime.now()
                # print(s2_paper.authors)
                # row.authors = s2_paper.authors
                # print(s2_paper)
                row.authors_num = len(s2_paper.authors) if s2_paper.authors is not None else None
            else:
                row.valid = 0
            session.commit()
    session.close()


# update_s2(session)
import json
from cfg.config import *
from sqlalchemy import and_
from retry import retry
from sqlalchemy.sql import text


@retry()
def update_s2_ref(session):


    results = session.query(PaperMapping).all()
    for row in tqdm(results):
        # if row.title == 'malaria likelihood prediction by effectively surveying households using deep reinforcement learning':
        #     pass
        if row.is_review == 1 and row.valid == 1:# row.reference_details is None and

            url = f'https://api.semanticscholar.org/graph/v1/paper/{row.s2_id}/references?fields=paperId,title,authors,intents,contexts,isInfluential,url,publicationVenue,openAccessPdf,abstract,venue,year,referenceCount,citationCount,influentialCitationCount,s2FieldsOfStudy,publicationTypes,journal,publicationDate&offset=0&limit=1000'

            with shelve.open(generate_cache_file_name(url)) as cache:
                # ref_count = row.s2_reference_count

                if url in cache:
                    # print('Cache loding')
                    reply = cache[url]
                    # print('Cache Loading Done')
                else:
                    # continue
                    req_session = requests.Session()
                    # s2api = None
                    if s2api is not None:
                        headers = {
                            'x-api-key': s2api
                        }
                    else:
                        headers = None
                    reply = req_session.get(url, headers=headers)
                    cache[url] = reply

                    time.sleep(0.33)
            try:
                response = reply.json()  # 0.15s
            except:
                continue
            # print('ssadad')

            if "data" not in response:
                row.valid = -1

            else:
                row.reference_details = json.dumps(response) #str(response['data'])
                session.commit()
    session.close()
    

@retry()
def update_s2_citation(session):
    results = session.query(CoP).all()

    for row in tqdm(results):
        if row.full_citation is None:
            OFFSET = 0
            citation_responses = []
            if True: #row.citation is None:
                while (True):
                    url = f'https://api.semanticscholar.org/graph/v1/paper/{row.s2_id}/citations?fields=paperId,title,venue,year,referenceCount,citationCount,publicationDate,publicationTypes&offset={OFFSET}&limit=1000'

                    with shelve.open(generate_cache_file_name(url)) as cache:
                        print(url)
                        if url in cache:
                            # print('Cache loding')
                            reply = cache[url]
                            # print('Cache Loading Done')
                        else:
                            time.sleep(1)
                            req_session = requests.Session()
                            # s2api = None
                            if s2api is not None:
                                headers = {
                                    'x-api-key': s2api
                                }
                            else:
                                headers = None
                            reply = req_session.get(url, headers=headers,verify=False)

                            cache[url] = reply

                        try:
                            response = reply.json()  # 0.15s
                        except:
                            OFFSET += 1000
                            continue

                        if "data" in response:
                            citation_responses.append(response)
                            if response['data'] == []:
                                break
                            OFFSET += 1000
                            # cache[url] = reply

                        else:
                            break


                row.full_citation = json.dumps(citation_responses)
                session.commit()

    session.close()


def insert_idLiterature_into_CoP(session):
    metadata = MetaData(bind=engine)
    table = Table('cop', metadata, autoload=True)
    delete_stmt = table.delete()
    session.execute(delete_stmt)
    session.commit()

    results = session.query(PaperMapping).all()
    for row in tqdm(results):
        if row.valid==1 and row.is_review ==1:
            try:
                cop_entry = CoP(row.s2_id, citation=None)
                session.add(cop_entry)
                session.commit()
            except SQLAlchemyError as e:

                session.rollback()
                row.valid = -1
                session.commit()
                print(e)
                # continue


import concurrent.futures
from tqdm import tqdm


def process_paper(row_id, session_factory):
    """Function to process each paper and check if it is a review"""
    # 使用 session_factory 创建一个独立的 session
    session = session_factory()

    # 查询单个 row 的数据
    row = session.query(PaperMapping).filter(PaperMapping.id == row_id).first()

    if row.is_review is None:
        status = check_PAMIreview(row.title, row.abstract)
        row.is_review = 1 if status else 0
        session.commit()

    result = f'{row.title}||{row.is_review}||{row.gpt_keywords}'

    session.close()  # 关闭 session
    return result


def check_is_review(session, session_factory):
    # 获取所有需要处理的论文记录
    results = session.query(PaperMapping).all()

    # 提取所有 row 的 id，避免在主线程中共享 session
    row_ids = [row.id for row in results]

    # 使用 ThreadPoolExecutor 并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        # 提交任务到线程池
        for row_id in tqdm(row_ids):
            future = executor.submit(process_paper, row_id, session_factory)
            futures.append(future)

        # 打印每个完成的任务结果
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    # 关闭主线程中的 session
    session.close()


# 运行函数

# update_s2_ref(session)


def sync_folder_to_database(session, dir=None):
    # Step 1 (optional): If data has already been injected during file download, this can be commented out (run arxiv downloader).
    # add_info_to_database(dir, session)
    
    # Step 2: Generate keywords and determine if it's a PAMI review.
    # check_is_review(session, session_factory)

    # Step 3: Query for Semantic Scholar (S2) information.
    print('Querying S2 information')
    update_s2(session, True)
    
    # Step 4: Update citation and reference information.
    # print('Updating citation and reference information')
    # update_s2_ref(session)
    # update_s2_citation(session)

    # Step 5 (optional): Update author information.
    # update_author(session)


def update_official_keywords(dir: str, session):
    import PyPDF2
    def extract_text_from_pdf(pdf_file_path):
        try:

            with open(pdf_file_path, 'rb') as pdf_file:

                pdf_reader = PyPDF2.PdfFileReader(pdf_file)

                if pdf_reader.numPages > 0:

                    page = pdf_reader.getPage(0)


                    text = page.extractText()

                    return text
                else:
                    return ''
        except Exception as e:
            return ''

    for filename in tqdm(os.listdir(dir)):
        if os.path.isfile(os.path.join(dir, filename)):
            pattern = r'\d+\.\d+(?:v\d+)?'
            match = re.search(pattern, filename)
            title_parts = filename.split('.')
            if match:

                # arxiv_id = match.group()
                arxiv_id = '.'.join(title_parts[:2])

                id = 'http://arxiv.org/abs/' + arxiv_id
                try:
                    data = session.query(PaperMapping).filter(PaperMapping.id == id).first()
                except Exception as e:
                    print(e)
                    continue
                # print(data.idLiterature)
                if data:
                    if data.keywords is None:
                        extracted_text = extract_text_from_pdf(os.path.join(dir, filename))
                        keywords = extract_keywords_from_article_with_gpt(extracted_text)
                        keywords = [keyword.replace('.', '').replace("'", "").replace('"', "") for keyword in keywords]
                        data.keywords = ';'.join(keywords)

                        if len(data.keywords) >= 255:
                            data.keywords = 'None'
                        try:
                            # print(data.keywords)
                            session.commit()
                        except Exception as e:
                            print(e)
                            continue

            else:
                print("No arXiv ID found.")


    session.close()
def update_metric(session):
    results = session.query(PaperMapping).filter(PaperMapping.is_review == 1).all()
    for row in tqdm(results):
        pass
def update_gpt_keyword(session):
    results = session.query(PaperMapping).all()
    for row in tqdm(results):
        # print(row.title)
        if row.gpt_keyword is None and row.is_review == 1 and row.valid == 1:
            keywords = get_chatgpt_field(row.title, row.abstract)
            keywords = [keyword.replace('.', '').replace("'", "").replace('"', "") for keyword in keywords]

            keyword = keywords[0]
            # row.gpt_keywords = ','.join(keywords)
            row.gpt_keyword = str(keyword)

            session.commit()

    session.close()

@retry()
def ensemble_meta_info(session):

    # results = session.query(PaperMapping).filter(PaperMapping.TNCSI == None and PaperMapping.gpt_keyword == None)#.all()#
    results = session.query(PaperMapping).filter(and_(PaperMapping.valid ==1, PaperMapping.is_review ==1, PaperMapping.TNCSI.is_(None), PaperMapping.CDR.is_(None))).all()
    for result in tqdm(tqdm(results)):
        s2paper = S2paper(result.title, filled_authors=False)
        if s2paper.citation_count is not None:  # S2bug
            try:
                result.gpt_keyword = s2paper.gpt_keyword
                result.TNCSI = s2paper.TNCSI['TNCSI']
                result.TNCSI_loc = s2paper.TNCSI['loc']
                result.TNCSI_scale = s2paper.TNCSI['scale']
                print(f"TNCSI: {result.TNCSI}", end=' | ')

                result.RQM = s2paper.RQM['RQM']
                result.ARQ = s2paper.RQM['ARQ']
                result.SMP = s2paper.RQM['S_mp']
                print(f"RQM: {result.RQM}", end=' | ')

                result.IEI = s2paper.IEI['L6']  if s2paper.IEI['L6']  != float('-inf') else None
                result.IEI_I6 = s2paper.IEI['I6']  if s2paper.IEI['I6']  != float('-inf') else None
                print(f"IEI: {result.IEI}", end=' | ')

                result.RUI = s2paper.RUI['RUI']
                result.RAD = s2paper.RUI['RAD']
                result.CDR = s2paper.RUI['CDR']
                print(f"RUI: {result.RUI}", end=' | ')

                session.commit()
                print("\nSuccessfully processed the paper.")
            except Exception as e:
                result.valid = -1
                session.commit()
        else:
            result.valid = 0
            session.commit()
if __name__ == "__main__":
    pass
    ensemble_meta_info(session)
    # update_gpt_keyword(session)
    # sync_folder_to_database(session,dir=r'J:\SLR')
    # update_official_keywords(r'E:/download_paper', session)
    # insert_idLiterature_into_CoP(session)
    # update_s2_citation(session)

