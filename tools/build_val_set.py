import pandas as pd

# Function to load the CSV and add required columns and headers
import os
import pandas as pd
from datetime import datetime
import os
import re
import shelve

import concurrent.futures
from tqdm import tqdm

# update_s2(session)
import json
from cfg.config import *
from sqlalchemy import and_
from retry import retry
from sqlalchemy.sql import text
import requests
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

from CACHE.CACHE_Config import generate_cache_file_name
from database.DBEntity import PaperMapping, CoP
from retrievers.Author import Author
from retrievers.arxiv_paper import Arxiv_paper

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from retrievers.google_scholar_paper import Google_paper
from retrievers.semantic_scholar_paper import S2paper
from tools.gpt_util import *
from cfg.safesession import session, engine, session_factory


# def prompt_eng_eval(session, file_path):
#     try:
#         # Load the CSV file using utf-8 encoding
#         df = pd.read_csv(file_path, encoding='utf-8')
#     except UnicodeDecodeError:
#         # If there's an encoding error, fall back to ISO-8859-1 encoding
#         df = pd.read_csv(file_path, encoding='ISO-8859-1')
#
#     # Initialize counters for correct predictions
#     correct_predictions = {
#         'taxonomy': 0,
#         'benchmark': 0,
#         'preliminary': 0,
#         'prisma': 0,
#         'application': 0,
#         'future': 0
#     }
#
#     total_rows = df.shape[0]
#
#     # Iterate through each row to update the columns
#     for index, row in tqdm(df.iterrows()):
#         title = row['title']
#
#         # Use the get_pdf_pth_by_title function to retrieve the rp object
#         try:
#             rp = get_review_feature_by_title(session, title)
#         except Exception as e:
#             print(e)
#             continue
#
#         # If rp object is found, compare values
#         if rp:
#             if rp.PROPOSE_TAXONOMY == row['taxonomy']:
#                 correct_predictions['taxonomy'] += 1
#             if rp.PROPOSE_BENCHMARK == row['benchmark']:
#                 correct_predictions['benchmark'] += 1
#             if rp.INCLUDE_PRELIMINARY == row['preliminary']:
#                 correct_predictions['preliminary'] += 1
#             if rp.USE_PRISMA == row['prisma']:
#                 correct_predictions['prisma'] += 1
#             if rp.INCLUDE_APPLICATION == row['application']:
#                 correct_predictions['application'] += 1
#             if rp.INCLUDE_FUTURE == row['future']:
#                 correct_predictions['future'] += 1
#
#     # Calculate overall accuracy
#     overall_accuracy = sum(correct_predictions.values()) / (total_rows * len(correct_predictions))
#
#     # Calculate individual category accuracies
#     category_accuracies = {key: value / total_rows for key, value in correct_predictions.items()}
#
#     # Create a DataFrame to display the results
#     results_df = pd.DataFrame({
#         'Category': list(category_accuracies.keys()) + ['Overall'],
#         'Accuracy': list(category_accuracies.values()) + [overall_accuracy]
#     })
#
#     # Print the results in a tabular format
#     print("\nAccuracy Results:")
#     print(results_df.to_string(index=False, float_format='%.2f'))
#
#     return overall_accuracy, category_accuracies

def single_review_feature_persistance(row_id, session_factory):
    """Function to process each paper and check if it is a review"""
    # 使用 session_factory 创建一个独立的 session
    session = session_factory()

    # 查询单个 row 的数据
    row = session.query(PaperMapping).filter(PaperMapping.id == row_id).first()

    if True:
        md_pth = os.path.join(r'J:\md\output',row.download_pth.replace('.pdf','.mmd'))
        if os.path.exists(md_pth):
            try:
                rp1 = get_taxonomy_criteria(md_pth, api_llm)
                rp2 = get_background_future_app(md_pth, api_llm)
                rp3 = get_benchmark(os.path.join(r'J:\SLR', row.download_pth), api_llm)

                rp = ReviewFeature()

                rp.PROPOSE_TAXONOMY = rp1.CLS_METHODS
                rp.INCLUDE_EXCLUSION_CRITERIA = rp1.SELECTION_CRITERIA

                rp.INCLUDE_PRELIMINARY = rp2.BACKGROUND  # INCLUDE_PRELIMINARY#background_app[0]
                rp.INCLUDE_FUTURE = rp2.DISSCUSSION
                rp.INCLUDE_APPLICATION = rp2.APPLICATION  # background_app[1]

                rp.INCLUDE_BENCHMARK = rp3.BENCHMARK
            except Exception as e:
                return None


            row.taxonomy = 1 if rp.PROPOSE_TAXONOMY else 0
            row.benchmark = 1 if  rp.INCLUDE_BENCHMARK else 0
            row.preliminary = 1 if rp.INCLUDE_PRELIMINARY else 0
            row.PRISMA = 1 if rp.INCLUDE_EXCLUSION_CRITERIA else 0
            row.application = 1 if rp.INCLUDE_APPLICATION else 0
            row.future = 1 if rp.INCLUDE_FUTURE else 0
            row.feature_analyzed =1
            session.commit()
        else:
            return None

    result = f'{row.title}||{row.is_review}'

    session.close()  # 关闭 session
    return result
from sqlalchemy import or_
def review_feature_persistance(session, session_factory):
    # 获取所有需要处理的论文记录
    results = (
        session.query(PaperMapping)
        .filter(
            PaperMapping.is_review == 1,
            PaperMapping.valid == -1,
            or_(PaperMapping.feature_analyzed != 1, PaperMapping.feature_analyzed.is_(None))
        )
        .all()
    )

    # 提取所有 row 的 id，避免在主线程中共享 session
    row_ids = [row.id for row in results]

    # 使用 ThreadPoolExecutor 并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        # 提交任务到线程池
        for row_id in tqdm(row_ids):
            future = executor.submit(single_review_feature_persistance, row_id, session_factory)
            futures.append(future)

        # 打印每个完成的任务结果
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    # 关闭主线程中的 session
    session.close()


def update_benchmark(pth, row, session):
    try:
        with open(pth, 'r', encoding='utf-8') as md_file:
            fig_tab_resp = fig_tab_extractor(pth)
            figs = fig_tab_resp.FIGURE_CAPTION
            tabs = fig_tab_resp.TABLE_CAPTION

            row.num_figs = len(figs)
            row.num_tabs = len(tabs)
            row.fig_caps = str(figs)
            row.tab_caps = str(tabs)
            session.commit()
    except Exception as e:
        return
from pdfminer.high_level import extract_pages, extract_text

def extract_pdf_text(pdf_path):
    try:
        # 提取 PDF 的全部文本
        text = extract_text(pdf_path)

        # 计算总页数
        total_pages = sum(1 for _ in extract_pages(pdf_path))

        return text, total_pages

    except Exception as e:
        print(f"An error occurred: {e}")
        return "", 0  # 返回空字符串和 0 页数
def count_words(pth, row, session):
    try:
        content,page_num =  extract_pdf_text(pth)
    # 使用 nltk 分词并计算单词数量
        words = nltk.word_tokenize(content)
        english_words = [word for word in words if word.isalpha()]  # 过滤非字母字符

        row.word_count = len(english_words)
        row.paper_num_pages = page_num
        # # 使用 split() 分割单词，并过滤出只包含字母的单词
        # str_words = [word for word in content.split() if word.isalpha()]

        # 计算两个列表之间的差异

        session.commit()

    except Exception as e:
        print(f"An error occurred: {e}")

def fig_tab_persistance(session,):
    # 获取所有需要处理的论文记录
    results = (
        session.query(PaperMapping)
        .filter(
            PaperMapping.feature_analyzed == 1,
            PaperMapping.valid== -1
        )
        .all()
    )

    # 提取所有 row 的 id，避免在主线程中共享 session
    rows = [row for row in results]

    # 使用 ThreadPoolExecutor 并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        # 提交任务到线程池
        for row in tqdm(rows):
            future = executor.submit(update_benchmark,os.path.join(r'J:\SLR', row.download_pth),row, session,)
            futures.append(future)

        # 打印每个完成的任务结果
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    # 关闭主线程中的 session
    session.close()
def stat_review_feature_by_status(session):
    results = session.query(PaperMapping).filter(PaperMapping.feature_analyzed == 1).all()

    # 初始化统计计数
    accepted_totals = {
        "taxonomy": 0,
        "benchmark": 0,
        "preliminary": 0,
        "PRISMA": 0,
        "application": 0,
        "future": 0,
    }

    unreviewed_totals = {
        "taxonomy": 0,
        "benchmark": 0,
        "preliminary": 0,
        "PRISMA": 0,
        "application": 0,
        "future": 0,
    }

    # 统计数量
    accepted_count = 0
    unreviewed_count = 0
    for row in results:
        if row.s2_pub_info == "arXiv.org":
            # 对未中稿的论文统计
            unreviewed_count += 1
            unreviewed_totals["taxonomy"] += 1 if row.taxonomy else 0
            unreviewed_totals["benchmark"] += 1 if row.benchmark else 0
            unreviewed_totals["preliminary"] += 1 if row.preliminary else 0
            unreviewed_totals["PRISMA"] += 1 if row.PRISMA else 0
            unreviewed_totals["application"] += 1 if row.application else 0
            unreviewed_totals["future"] += 1 if row.future else 0
        else:
            # 对已中稿的论文统计
            accepted_count += 1
            accepted_totals["taxonomy"] += 1 if row.taxonomy else 0
            accepted_totals["benchmark"] += 1 if row.benchmark else 0
            accepted_totals["preliminary"] += 1 if row.preliminary else 0
            accepted_totals["PRISMA"] += 1 if row.PRISMA else 0
            accepted_totals["application"] += 1 if row.application else 0
            accepted_totals["future"] += 1 if row.future else 0

    # 计算比例
    accepted_proportions = {feature: (count / accepted_count) if accepted_count else 0 for feature, count in accepted_totals.items()}
    unreviewed_proportions = {feature: (count / unreviewed_count) if unreviewed_count else 0 for feature, count in unreviewed_totals.items()}

    # 打印数量结果
    print("Accepted Papers Feature Counts:")
    for feature, count in accepted_totals.items():
        print(f"{feature}: {count}")

    print("\nUnreviewed Papers Feature Counts:")
    for feature, count in unreviewed_totals.items():
        print(f"{feature}: {count}")

    # 返回比例结果
    return accepted_proportions, unreviewed_proportions

import matplotlib.pyplot as plt
from collections import defaultdict

def stat_review_feature(session):
    results = session.query(PaperMapping).filter(PaperMapping.feature_analyzed == 1).all()
    all_totals_by_year = defaultdict(lambda: {
        "taxonomy": 0,
        "benchmark": 0,
        "preliminary": 0,
        "PRISMA": 0,
        "application": 0,
        "future": 0,
        "count": 0
    })

    # Initialize data structures to store counts by year
    accepted_totals_by_year = defaultdict(lambda: {
        "taxonomy": 0,
        "benchmark": 0,
        "preliminary": 0,
        "PRISMA": 0,
        "application": 0,
        "future": 0,
        "count": 0
    })

    unreviewed_totals_by_year = defaultdict(lambda: {
        "taxonomy": 0,
        "benchmark": 0,
        "preliminary": 0,
        "PRISMA": 0,
        "application": 0,
        "future": 0,
        "count": 0
    })

    # Process results and populate totals by year
    for row in results:
        year = row.publication_date.year  # Extract the year from publication_date


        all_totals_by_year[year]["count"] += 1
        all_totals_by_year[year]["taxonomy"] += 1 if row.taxonomy else 0
        all_totals_by_year[year]["benchmark"] += 1 if row.benchmark else 0
        all_totals_by_year[year]["preliminary"] += 1 if row.preliminary else 0
        all_totals_by_year[year]["PRISMA"] += 1 if row.PRISMA else 0
        all_totals_by_year[year]["application"] += 1 if row.application else 0
        all_totals_by_year[year]["future"] += 1 if row.future else 0
        if row.s2_pub_info == "arXiv.org":
            # Unreviewed papers
            unreviewed_totals_by_year[year]["count"] += 1
            unreviewed_totals_by_year[year]["taxonomy"] += 1 if row.taxonomy else 0
            unreviewed_totals_by_year[year]["benchmark"] += 1 if row.benchmark else 0
            unreviewed_totals_by_year[year]["preliminary"] += 1 if row.preliminary else 0
            unreviewed_totals_by_year[year]["PRISMA"] += 1 if row.PRISMA else 0
            unreviewed_totals_by_year[year]["application"] += 1 if row.application else 0
            unreviewed_totals_by_year[year]["future"] += 1 if row.future else 0
        else:
            # Accepted papers
            accepted_totals_by_year[year]["count"] += 1
            accepted_totals_by_year[year]["taxonomy"] += 1 if row.taxonomy else 0
            accepted_totals_by_year[year]["benchmark"] += 1 if row.benchmark else 0
            accepted_totals_by_year[year]["preliminary"] += 1 if row.preliminary else 0
            accepted_totals_by_year[year]["PRISMA"] += 1 if row.PRISMA else 0
            accepted_totals_by_year[year]["application"] += 1 if row.application else 0
            accepted_totals_by_year[year]["future"] += 1 if row.future else 0

    # Calculate proportions by year
    def calculate_proportions(totals_by_year):
        proportions_by_year = defaultdict(dict)
        for year, totals in totals_by_year.items():
            count = totals.pop("count", 1)  # Avoid division by zero
            for feature, feature_count in totals.items():
                proportions_by_year[year][feature] = feature_count / count
        return proportions_by_year

    accepted_proportions_by_year = calculate_proportions(accepted_totals_by_year)
    unreviewed_proportions_by_year = calculate_proportions(unreviewed_totals_by_year)
    all_proportions_by_year = calculate_proportions(all_totals_by_year)

    # Plot feature trends over years
    def plot_feature_trends(proportions_by_year, title):
        features = ["taxonomy", "benchmark", "preliminary", "PRISMA", "application", "future"]
        years = sorted(proportions_by_year.keys())

        plt.figure(figsize=(12, 8))
        for feature in features:
            plt.plot(years, [proportions_by_year[year].get(feature, 0) for year in years], label=feature)

        plt.xlabel("Year")
        plt.ylabel("Proportion")
        plt.title(title)
        plt.legend()
        plt.show()

    # Plot accepted and unreviewed trends
    # plot_feature_trends(accepted_proportions_by_year, "Accepted Papers Feature Proportions Over Years")
    # plot_feature_trends(unreviewed_proportions_by_year, "Unreviewed Papers Feature Proportions Over Years")

    # Return the calculated proportions by year
    return accepted_proportions_by_year, unreviewed_proportions_by_year, all_proportions_by_year


api_llm = 'gpt-4o-mini'

def get_review_feature_by_title(session,title):
    # 获取所有需要处理的论文记录
    result = session.query(PaperMapping).filter(PaperMapping.title==title).first()

    print()
    md_pth = os.path.join(r'J:\md\output',result.download_pth.replace('.pdf','.mmd'))
    if False:
        rp = get_review_feature_full(md_pth, 'gpt-4o-mini')
    else:
        rp1 = get_taxonomy_criteria(md_pth,api_llm)
        rp2 = get_background_future_app(md_pth, api_llm)
        rp3 = get_benchmark(os.path.join(r'J:\SLR',result.download_pth), api_llm)
        # background_app = get_review_feature_sub3_agent(md_pth,'gpt-4o-mini')

        # rp = get_review_feature_full(md_pth,'gpt-4o-mini')
        # rp.PROPOSE_TAXONOMY
        # rp.PROPOSE_BENCHMARK
        # rp.PROPOSE_NEW_METHOD
        # rp.USE_PRISMA


        rp = ReviewFeature()

        rp.PROPOSE_TAXONOMY = rp1.CLS_METHODS
        rp.INCLUDE_EXCLUSION_CRITERIA = rp1.SELECTION_CRITERIA


        rp.INCLUDE_PRELIMINARY = rp2.BACKGROUND  # INCLUDE_PRELIMINARY#background_app[0]
        rp.INCLUDE_FUTURE = rp2.DISSCUSSION
        rp.INCLUDE_APPLICATION = rp2.APPLICATION  # background_app[1]

        rp.INCLUDE_BENCHMARK = rp3.BENCHMARK

    return rp

# Function to get 'rp' object attributes by title and update DataFrame
def build_keyword_GT(session, file_path):

    try:
        # Try to load the CSV file using utf-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If there's an encoding error, fall back to ISO-8859-1 encoding
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Ensure that the necessary columns are present
    if 'propose_taxonomy' not in df.columns:
        df['propose_taxonomy'] = None
    if 'propose_benchmark' not in df.columns:
        df['propose_method'] = None
    if 'propose_new_method' not in df.columns:
        df['propose_new_method'] = None
    if 'use_prisma' not in df.columns:
        df['use_prisma'] = None

    # Iterate through each row to update the columns
    for index, row in tqdm(df.iterrows()):
        title = row['title']

        # Use the get_pdf_pth_by_title function to retrieve the rp object
        try:
            rp = get_review_feature_by_title(session, title)
        except Exception as e:
            print(e)
            continue
        # If rp object is found, update the corresponding columns
        if rp:
            df.at[index, 'propose_taxonomy'] = rp.PROPOSE_TAXONOMY
            df.at[index, 'propose_benchmark'] = rp.PROPOSE_BENCHMARK
            df.at[index, 'propose_new_method'] = rp.PROPOSE_NEW_METHOD
            df.at[index, 'use_prisma'] = rp.USE_PRISMA

        # Save the updated DataFrame back to the same CSV file
        df.to_csv(file_path, index=False, encoding='utf-8')

    return df


def filter_csv(session, file_path):
    try:
        # Try to load the CSV file using utf-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If there's an encoding error, fall back to ISO-8859-1 encoding
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Filter rows where 'propose_taxonomy' is not a boolean
    total_before = len(df)
    df = df[df['propose_taxonomy'].apply(lambda x: isinstance(x, bool))]
    total_after = len(df)

    # Save the updated DataFrame back to the same CSV file
    df.to_csv(file_path, index=False, encoding='utf-8')

    # Output the number of rows filtered
    total_filtered = total_before - total_after
    print(f"Total rows before filtering: {total_before}")
    print(f"Total rows after filtering: {total_after}")
    print(f"Total rows filtered out: {total_filtered}")

    return total_filtered

import pandas as pd
from tqdm import tqdm

def inject_arxiv_id(session, file_path):
    try:
        # Try to load the CSV file using utf-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If there's an encoding error, fall back to ISO-8859-1 encoding
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Add 'arxiv_id' column if it does not exist
    if 'arxiv_id' not in df.columns:
        df['arxiv_id'] = None

    # Iterate through each row to update the 'arxiv_id' column
    for index, row in tqdm(df.iterrows()):
        title = row['title']
        result = session.query(PaperMapping).filter(PaperMapping.title == title).first()

        if result:
            arxiv_id = '.'.join(result.download_pth.split('.')[:2])
            # Update the DataFrame directly
            df.at[index, 'arxiv_id'] = arxiv_id

    # Save the updated DataFrame back to the same CSV file
    df.to_csv(file_path, index=False, encoding='utf-8')

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def prompt_eng_eval( file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    # 只取最后50行
    # df = df.tail(50)
    # df = df.tail(25)

    correct_predictions = {
        'taxonomy': 0,
        'benchmark': 0,
        'preliminary': 0,
        'prisma': 0,
        'application': 0,
        'future': 0
    }

    total_rows = 0
    title_to_row = {row['title']: row for index, row in df.iterrows()}

    # 存储错误的ID和错误的结果
    incorrect_predictions = {
        'taxonomy': [],
        'benchmark': [],
        'preliminary': [],
        'prisma': [],
        'application': [],
        'future': []
    }

    from sqlalchemy.orm import scoped_session, sessionmaker

    def fetch_rp(title):
        session = session_factory()  # 为每个线程创建新的 session
        try:
            return title, get_review_feature_by_title(session, title)
        except Exception as e:
            print(e)
            return title, None
        finally:
            session.close()  # 确保 session 在任务结束后关闭

    with ThreadPoolExecutor(max_workers=8) as executor: #8
        futures = {executor.submit(fetch_rp, title): title for title in title_to_row.keys()}

        for future in tqdm(as_completed(futures)):
            title, rp = future.result()
            if rp is not None and title in title_to_row:
                row = title_to_row[title]
                if rp.PROPOSE_TAXONOMY == row['taxonomy']:
                    correct_predictions['taxonomy'] += 1
                else:
                    incorrect_predictions['taxonomy'].append((title, rp.PROPOSE_TAXONOMY, row['taxonomy']))

                if rp.INCLUDE_BENCHMARK == row['benchmark']:
                    correct_predictions['benchmark'] += 1
                else:
                    incorrect_predictions['benchmark'].append((title, rp.INCLUDE_BENCHMARK, row['benchmark']))

                if rp.INCLUDE_PRELIMINARY == row['preliminary']:
                    correct_predictions['preliminary'] += 1
                else:
                    incorrect_predictions['preliminary'].append((title, rp.INCLUDE_PRELIMINARY, row['preliminary']))

                if rp.INCLUDE_EXCLUSION_CRITERIA == row['prisma']:
                    correct_predictions['prisma'] += 1
                else:
                    incorrect_predictions['prisma'].append((title, rp.INCLUDE_EXCLUSION_CRITERIA, row['prisma']))

                if rp.INCLUDE_APPLICATION == row['application']:
                    correct_predictions['application'] += 1
                else:
                    incorrect_predictions['application'].append((title, rp.INCLUDE_APPLICATION, row['application']))

                if rp.INCLUDE_FUTURE == row['future']:
                    correct_predictions['future'] += 1
                else:
                    incorrect_predictions['future'].append((title, rp.INCLUDE_FUTURE, row['future']))

                total_rows += 1

    overall_accuracy = sum(correct_predictions.values()) / (total_rows * len(correct_predictions))
    category_accuracies = {key: value / total_rows for key, value in correct_predictions.items()}

    results_df = pd.DataFrame({
        'Category': list(category_accuracies.keys()) + ['Overall'],
        'Accuracy': list(category_accuracies.values()) + [overall_accuracy]
    })

    print("\nAccuracy Results:")
    print(results_df.to_string(index=False, float_format='%.2f'))

    # 输出错误的ID和结果
    print("\nIncorrect Predictions:")
    for category, errors in incorrect_predictions.items():
        if errors:
            print(f"\nCategory: {category}")
            for title, predicted, actual in errors:
                print(f"Title: {title}, Predicted: {predicted}, Actual: {actual}")

    return overall_accuracy, category_accuracies


import concurrent.futures
import pandas as pd
from tqdm import tqdm




if __name__ == '__main__':
    file_path = r'C:\Users\Ocean\Documents\GitHub\Dynamic_Literature_Review\test_unit\ReviewFeature-GT-mini.csv'  # Replace with actual file path
    # print()
    acc_p, ur_p = stat_review_feature(session)
    print(acc_p)
    print(ur_p)
    # inject_arxiv_id(session, file_path)
    # prompt_eng_eval( file_path)
    # review_feature_persistance(session, session_factory)
    # import nltk
    #
    # nltk.download('punkt_tab')
    #
    # fig_tab_persistance(session)
    # ar, ur = stat_review_feature_by_status(session)
    # print("Accept:", ar)
    # print("Under review:", ur)