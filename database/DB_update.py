import shelve
import requests
from CACHE.cache_request import generate_cache_file_name
from database.DBEntity import PaperMapping, CoP
from retrievers.arxiv_paper import Arxiv_paper
import concurrent.futures
from retrievers.semantic_scholar_paper import S2paper
from tools.gpt_util import *
from cfg.safesession import *
import json
from cfg.config import *
from sqlalchemy import and_
from retry import retry
from tqdm import tqdm
import time
import datetime



def add_info_to_database(dir: str, session):
    '''
    arxiv下载后论文的第一步
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



def update_s2(session, mandatory_update=False):
    results = session.query(PaperMapping).filter(PaperMapping.s2_citation_count.is_(None)).all()
    for row in tqdm(results):
        if (row.s2_id is None or mandatory_update) and row.is_review == 1:
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
                row.last_update_time = datetime.datetime.now()
                # print(s2_paper.authors)
                # row.authors = s2_paper.authors
                # print(s2_paper)
                row.authors_num = len(s2_paper.authors) if s2_paper.authors is not None else None
            else:
                row.valid = 0
            session.commit()
    session.close()


@retry()
def update_s2_ref(session):
    results = session.query(PaperMapping).all()
    for row in tqdm(results):
        # if row.title == 'malaria likelihood prediction by effectively surveying households using deep reinforcement learning':
        #     pass
        if row.is_review == 1 and row.valid == 1:  # row.reference_details is None and

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
                row.reference_details = json.dumps(response)  # str(response['data'])
                session.commit()
    session.close()


@retry()
def update_s2_citation(session):
    results = session.query(CoP).all()

    for row in tqdm(results):
        if row.full_citation is None:
            OFFSET = 0
            citation_responses = []
            if True:  # row.citation is None:
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
                            reply = req_session.get(url, headers=headers, verify=False)

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



def process_paper_is_review(row_id, session_factory):
    """Function to process each paper and check if it is a review"""
    # 
    session = session_factory()

    # 
    row = session.query(PaperMapping).filter(PaperMapping.id == row_id).first()

    if row.is_review is None:
        status = check_PAMIreview(row.title, row.abstract)
        row.is_review = 1 if status else 0
        session.commit()

    result = f'{row.title}||{row.is_review}||{row.gpt_keywords}'

    session.close()  # 
    return result
def check_is_review(session, session_factory):
    # 
    results = session.query(PaperMapping).all()

    # 
    row_ids = [row.id for row in results]

    # 
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        # 
        for row_id in tqdm(row_ids):
            future = executor.submit(process_paper_is_review, row_id, session_factory)
            futures.append(future)

        # 
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    # 
    session.close()




def update_official_keywords(dir: str, session):
    '''
    extract keywords provide by authors to database. If there exist one...
    '''
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




def limit_precision(value, threshold=1e-8):
    value = float(value)
    return 0.0 if abs(value) < threshold else value


@retry()
def update_indicators_value(session):
    '''
       update indicators in database
    '''
    results = session.query(PaperMapping).filter(
        and_(PaperMapping.is_review == 1, PaperMapping.TNCSI._is(None))).all()
    for result in tqdm(tqdm(results)):
        s2paper = S2paper(result.title, filled_authors=False)
        if s2paper.citation_count is not None:  # S2bug
            try:
                result.gpt_keyword = s2paper.gpt_keyword

                result.TNCSI = limit_precision(f"{s2paper.TNCSI['TNCSI']:.4e}")
                result.TNCSI_loc = s2paper.TNCSI['loc']
                result.TNCSI_scale = s2paper.TNCSI['scale']
                print(f"TNCSI: {result.TNCSI}", end=' | ')

                result.RQM = float(f"{s2paper.RQM['RQM']:.4e}")
                result.ARQ = limit_precision(f"{s2paper.RQM['ARQ']:.4e}")
                result.SMP = s2paper.RQM['S_mp']
                print(f"RQM: {result.RQM}", end=' | ')

                result.IEI = s2paper.IEI['L6'] if s2paper.IEI['L6'] != float('-inf') else None
                result.IEI_I6 = s2paper.IEI['I6'] if s2paper.IEI['I6'] != float('-inf') else None
                print(f"IEI: {result.IEI}", end=' | ')

                result.RUI = limit_precision(f"{s2paper.RUI['RUI']:.4e}") if s2paper.RUI['RUI'] != float(
                    '-inf') else None
                result.RAD = limit_precision(f"{s2paper.RUI['RAD']:.4e}") if s2paper.RUI['RAD'] != float(
                    '-inf') else None
                result.CDR = limit_precision(f"{s2paper.RUI['CDR']:.4e}") if s2paper.RUI['CDR'] != float(
                    '-inf') else None
                print(f"RUI: {result.RUI}", end=' | ')

                session.commit()
                print("\nSuccessfully processed the paper.")
            except Exception as e:

                print(e)
                result.valid = -1
                session.commit()
        else:
            result.valid = 0
            session.commit()

def sync_folder_to_database(session, dir=None):
    # Step 1 (optional): If data has already been injected during file download, this can be commented out (run arxiv downloader).
    add_info_to_database(dir, session)

    # Step 2: Generate keywords and determine if it's a PAMI review.
    check_is_review(session, session_factory)

    # Step 3: Query for Semantic Scholar (S2) information.
    print('Querying S2 information')
    update_s2(session, True)

    # Step 4: Update citation and reference information.
    print('Updating citation and reference information')
    update_s2_ref(session)
    update_s2_citation(session)



if __name__ == "__main__":
    pass
    # sync_folder_to_database(session, dir=None)


    # Assuming file_path is the location of your CSV file

    # df_with_headers = build_keyword_GT(session, file_path)
    # ensemble_meta_info(session)
    # update_gpt_keyword(session)
    # sync_folder_to_database(session,dir=r'J:\SLR')
    # update_official_keywords(r'E:/download_paper', session)
    # insert_idLiterature_into_CoP(session)
    # update_s2_citation(session)

