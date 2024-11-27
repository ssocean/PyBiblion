import os
import re
from tqdm import tqdm
from database.DBEntity import PaperMapping
from retrievers.arxiv_paper import Arxiv_paper
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from retrievers.arxiv_paper import Arxiv_paper
from retrievers.google_scholar_paper import Google_paper
from tools.gpt_util import *
import time
import random

def get_valid_rows(session):
    valid_rows = []
    results = session.query(PaperMapping).all()


    for row in tqdm(results):
        if row.is_review == 1 and row.valid == 1:
            # if row.RUI and row.IEI and row.RAD and row.CDR:
            valid_rows.append(row)
    return valid_rows

def statistics_database(session):
    rst = {}
    results = session.query(PaperMapping).all()
    years = []
    ref_counts = []
    citation_counts = []
    pubs = []
    authors_num = []

    TNCSIs = []
    RQMs = []
    ARQs = []
    SMPs = []
    RUIs = []
    IEIs = []
    kwds = []
    RADs = []
    CDRs = []

    for row in tqdm(results):
        if row.is_review == 1 and row.valid == 1:
            # if row.RUI and row.IEI and row.RAD and row.CDR:
            years.append(int(row.publication_date.year))
            ref_counts.append(int(row.s2_reference_count))
            citation_counts.append(int(row.s2_citation_count))
            pubs.append(row.s2_pub_info)
            authors_num.append(row.authors_num)

            # Now append to the new lists
            TNCSIs.append(row.TNCSI)  # Assuming row has a field named TNCSI
            RQMs.append(row.RQM)  # Assuming row has a field named RQM
            ARQs.append(row.ARQ)  # Assuming row has a field named ARQ
            SMPs.append(row.SMP)  # Assuming row has a field named SMP
            RUIs.append(row.RUI)  # Assuming row has a field named RUI
            IEIs.append(row.IEI)  # Assuming row has a field named IEI
            kwds.append(row.gpt_keyword)  # Assuming row has a field named kwds
            RADs.append(row.RAD)  # Assuming row has a field named RAD
            CDRs.append(row.CDR)  # Assuming row has a field named CDR

    # Adding all the lists to the result dictionary
    rst['years'] = years
    rst['ref_counts'] = ref_counts
    rst['citation_counts'] = citation_counts
    rst['pubs'] = pubs
    rst['authors_num'] = authors_num
    rst['TNCSIs'] = TNCSIs
    rst['RQMs'] = RQMs
    rst['ARQs'] = ARQs
    rst['SMPs'] = SMPs
    rst['RUIs'] = RUIs
    rst['IEIs'] = IEIs
    rst['kwds'] = kwds
    rst['RADs'] = RADs
    rst['CDRs'] = CDRs

    return rst
