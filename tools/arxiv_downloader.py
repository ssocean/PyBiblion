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



# Multi Process SAFE SESSION
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session_factory = scoped_session(SessionLocal)  # 使用 scoped_session

def replace_invalid_characters(path):
    pattern = r'[<>:"/\\|?*]'
    new_path = re.sub(pattern, '_', path)

    return new_path


def auto_make_directory(dir_pth: str):
    if os.path.exists(dir_pth):  
        return True
    else:
        os.makedirs(dir_pth)
        return False


def init_logger(out_pth: str = 'logs'):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    auto_make_directory(out_pth)
    handler = logging.FileHandler(fr'{out_pth}/{time.strftime("%Y_%b_%d", time.localtime())}_log.txt')

    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s # %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
logger = init_logger()


def download_arxiv_paper(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print("Download Success！")
    else:
        print("Download Failed！")


import os


def file_exists_case_insensitive(id, title, files, extension='pdf'):
    clean_title = '_'.join(re.findall(r'\w+', title))
    file_name_with_suffix = "{}.{}.{}".format(id, clean_title, extension)
    return file_name_with_suffix in files


@retry()
def redownload_single_process(session,out_dir):
    files = [f.lower() for f in os.listdir(out_dir)]
    results = session.query(PaperMapping).all()
    error_list = []
    for row in tqdm(results):
        if row.has_been_downloaded == 0:
            pub_url = row.pub_url
            id = get_arxiv_id_from_url(pub_url)

            file_pth = os.path.join(out_dir, id + '.' + row.title + '.pdf')

            # print(file_pth)
            if not file_exists_case_insensitive(id, row.title,
                                                files=files):  # os.path.exists(os.path.join(out_dir, arxiv_paper._get_default_filename())):
                search = arxiv.Search(id_list=[id]).results()
                arxiv_paper = search.__next__()
                logger.info(
                    f'{arxiv_paper._get_default_filename()} is not existed in the disk, try downloading it.')
                # arxiv_paper.download_pdf(dirpath=out_dir)
                try:
                    save_path = os.path.join(out_dir, arxiv_paper._get_default_filename())
                    arxiv_paper.download_pdf(dirpath=out_dir)
                    row.has_been_downloaded = 1
                    session.commit()

                except URLError:
                    time.sleep(8)
                    logger.warning(f'{arxiv_paper._get_default_filename()} download failed.')
                    error_list.append(arxiv_paper._get_default_filename())
            else:
                row.has_been_downloaded = 1
                session.commit()

    session.close()

@retry(delay=8)
def download_paper(row, out_dir):
    session = session_factory()
    if row.downloaded_pth is None:
        pub_url = row.pub_url
        id = get_arxiv_id_from_url(pub_url)

        try:
            search = arxiv.Search(id_list=[id]).results()
            arxiv_paper = next(search)
            save_path = os.path.join(out_dir, arxiv_paper._get_default_filename())
            arxiv_paper.download_pdf(dirpath=out_dir)
            row.downloaded_pth = arxiv_paper._get_default_filename()
            row.paper_type = 1
            session.commit()
            return f"{arxiv_paper._get_default_filename()} downloaded successfully."
        except URLError:
            session.rollback()

            return f"{arxiv_paper._get_default_filename()} download failed."
        except Exception as e:
            session.rollback()

            return f"Error downloading {arxiv_paper._get_default_filename()}: {str(e)}"
        finally:
            session.close()  # 关闭 session
@retry(delay=6)
def redownload(out_dir):
    session = session_factory()
    results = session.query(PaperMapping).all() # you can add .filter(is_downloaded=0) to only download new papers
    session.close()  # 关闭 session
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(download_paper, row, out_dir) for row in results]
        for future in tqdm(as_completed(futures), total=len(futures)):
            logger.info(future.result())


def simple_query():
    query = f'all:"infrared" AND all:"small" AND all:"Target detection"'

    search = arxiv.Search(query=query,
                          max_results=float('inf'),
                          sort_by=SortCriterion.Relevance,
                          sort_order=SortOrder.Descending,
                          )

    for i, result in enumerate(search.results()):
        print(i, result.title, result.entry_id, result.published, )
        print(result.summary)
        print('\n\n')


@retry(delay=16)
def main(session):
    # add or remove kwds if you want
    key_words = ["Action Detection", "Action Recognition", "Activity Detection", "Activity Recognition", "Adversarial Attack", "Anomaly Detection", "Audio Classification", "Biometric Authentication", "Biometric Identification", "Boundary Detection", "CNN", "Computer Vision", "Contrastive Learning", "Data Mining", "Data Visualization", "Depth Estimation", "Dialogue Modeling", "Dialogue Systems", "Diffusion Model", "Document Analysis", "Document Analysis and Recognition", "Document Clustering", "Document Layout Analysis", "Document Retrieval", "Domain Adaptation", "Edge Detection", "Emotion Recognition", "Facial Recognition", "Face Detection", "Face Recognition", "Gesture Analysis", "Gesture Recognition", "Graph Mining", "Hand Gesture Recognition", "Handwriting Recognition", "Human Activity Recognition", "Human Detection", "Human Pose Estimation", "Image Captioning", "Image Classification", "Image Clustering", "Image Compression", "Image Editing", "Image Enhancement", "Image Generation", "Image Inpainting", "Image Matching", "Image Quality Assessment", "Image Recognition", "Image Reconstruction", "Image Retrieval", "Image Restoration", "Image Segmentation", "Image-Based Localization", "Instance Segmentation", "Knowledge Graph", "Knowledge Representation", "Language Modeling", "Language Modelling", "Machine Learning Interpretability", "Machine Translation", "Medical Image Analysis", "Medical Image Segmentation", "Meta-Learning", "Metric Learning", "Multi-Label Classification", "Named Entity Disambiguation", "Named Entity Recognition", "Natural Language Processing", "Object Detection", "Object Tracking", "Optical Character Recognition", "Pattern Matching", "Pattern Recognition", "Person Re-Identification", "Point Cloud", "Question Answering", "Recommendation Systems", "Recommender Systems", "Relation Extraction", "Remote Sensing", "Representation Learning", "Saliency Detection", "Salient Object Detection", "Scene Segmentation", "Scene Understanding", "Semantic Segmentation", "Sentiment Analysis", "Sentiment Classification", "Signature Verification", "Speech Emotion Recognition", "Speech Enhancement", "Speech Recognition", "Speech Synthesis", "Speech-to-Text Conversion", "Super-Resolution", "Superpixels", "Text Classification", "Text Clustering", "Text Generation", "Text Mining", "Text Summarization", "Text-to-Image Generation", "Text-to-Speech Conversion", "Text-to-Speech Synthesis", "Time Series Analysis", "Time Series Forecasting", "Topic Detection", "Topic Modeling", "Transfer Learning", "Video Object Segmentation", "Video Processing", "Video Summarization", "Video Understanding", "Visual Question Answering", "Visual Tracking", "Word Embeddings", "Zero-Shot Learning"]
    key_words = [i.lower() for i in key_words]

    key_words = list(set(key_words))
    logger.info(key_words)
    logger.info(len(key_words))
    key_words_count = {}
    error_list = []

    for key_word in tqdm(key_words):
        # Make your own search rules here. Check https://info.arxiv.org/help/api/user-manual.html#query_details for more infomation.
        # More Examples: query = 'abs:"CLIP" AND abs:"knowledge distillation"'
        query = f'(ti:"review" OR ti:"survey") AND abs:"{key_word.lower()}"'

        logger.info(f'Start query {key_word}')
        search = arxiv.Search(query=query,
                              max_results=float('inf'),
                              sort_by=SortCriterion.Relevance,
                              sort_order=SortOrder.Descending,
                              )

        for i, result in enumerate(search.results()):
            if '/' in result._get_default_filename():
                if '\\' in result._get_default_filename():
                    continue
                continue

            if True:  # result.published.year>=2022:
                print(result._get_default_filename() + ' retrived earlier. Skipping now.')
                arxiv_paper = Arxiv_paper(result, ref_type='entity')
                data = session.query(PaperMapping).filter(PaperMapping.id == arxiv_paper.id).first()

                if data is None:  # current paper is not in the database
                    if not os.path.exists(os.path.join(out_dir, result._get_default_filename())):
                        logger.info(f'{arxiv_paper.id} is not existed in the disk, try downloading it. ')
                        try:
                            result.download_pdf(dirpath=out_dir)
                            pass
                        except URLError:
                            time.sleep(8)
                            logger.warning(f'{result._get_default_filename()} download failed.')
                            error_list.append(result._get_default_filename())

                    doc = PaperMapping(arxiv_paper=arxiv_paper, search_by_keywords=query)
                    session.add(doc)
                    session.commit()

                else:
                    data.search_by_keywords = query
                    session.commit()

                if key_word.lower() not in key_words_count:
                    key_words_count[f'{key_word.lower()}'] = 1
                else:
                    key_words_count[f'{key_word.lower()}'] += 1

                # print(str(i) + ' ' + result._get_default_filename())

    session.close()
    print(key_words_count)
    import json
    with open("kwd_couont.json", "w") as json_file:
        json.dump(key_words_count, json_file)
    for err in error_list:
        print(err)


if __name__ == "__main__":
    out_dir = r"C:\download_paper"
    main(session)
    
    # session.expunge_all()
    # redownload(session)
    # simple_query()
