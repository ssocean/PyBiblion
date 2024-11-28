import uuid
from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from retrievers.arxiv_paper import Arxiv_paper
from retrievers.semantic_scholar_paper import S2paper

Base = declarative_base()


class PaperMapping(Base):
    __tablename__ = 'literature'  

    idLiterature = Column(String(36), primary_key=True)
    title = Column(String(255))
    publication_date = Column(DateTime, default=None)
    language = Column(String(20), default='eng')
    id = Column(String(45))
    publisher = Column(String(255))
    publication_source = Column(String(255))
    source_type = Column(String(255))
    keywords = Column(String(255))
    abstract = Column(Text)
    citation_count = Column(Integer)
    references = Column(Text)
    pub_url = Column(String(255))
    comment = Column(Text)
    journal_ref = Column(String(255))
    authors = Column(Text)
    gpt_keywords = Column(String(255))
    download_pth = Column(Text)
    search_by_keywords = Column(String(255))
    aop = Column(String(36))
    # primary_category= Column(String(45))
    # categories= Column(String(45))
    # links = Column(String(255))
    s2_id = Column(String(45))
    s2_publication_date = Column(DateTime, default=None)
    s2_tldr = Column(Text)
    s2_DOI = Column(String(255))
    s2_pub_info = Column(Text)
    s2_pub_url = Column(String(255))
    s2_citation_count = Column(Integer)
    s2_reference_count = Column(Integer)
    s2_field = Column(String(255))
    s2_influential_citation_count = Column(Integer)
    valid = Column(Integer)
    is_review = Column(Integer)
    last_update_time = Column(DateTime, default=datetime.now())
    reference_details = Column(Text)
    authors_num = Column(Integer)
    has_been_downloaded = Column(Integer)
    gpt_keyword = Column(String(255))
    TNCSI = Column(Float)
    TNCSI_S = Column(Float)
    RQM = Column(Float)
    ARQ = Column(Float)
    SMP = Column(Float)
    RUI = Column(Float)
    IEI = Column(Float)
    IEI_I6 = Column(Float)
    IEI_norm = Column(Float)
    IEI_I6_norm = Column(Float)
    TNCSI_loc = Column(Float)
    TNCSI_scale = Column(Float)
    CDR = Column(Float)
    RAD = Column(Float)
    page_count = Column(Integer)
    word_count = Column(Integer)
    review_type = Column(String(255))
    taxonomy = Column(Integer)
    benchmark = Column(Integer)
    preliminary    = Column(Integer)
    PRISMA = Column(Integer)
    future = Column(Integer)
    application = Column(Integer)
    feature_analyzed = Column(Integer)
    paper_num_pages = Column(Integer)
    num_figs = Column(Integer)
    num_tabs = Column(Integer)
    fig_caps = Column(Text)
    tab_caps = Column(Text)
    def __init__(self, arxiv_paper: Arxiv_paper = None, s2_paper: S2paper = None, search_by_keywords=None):
        if arxiv_paper is not None:
            # 
            idLiterature = uuid.uuid4()

            # 
            self.idLiterature = str(idLiterature)
            self.title = arxiv_paper.title
            self.publication_date = arxiv_paper.publication_date

            self.id = arxiv_paper.id

            self.publication_source = arxiv_paper.publication_source

            self.keywords = arxiv_paper.keywords
            self.abstract = arxiv_paper.abstract
            self.citation_count = arxiv_paper.citation_count

            self.pub_url = arxiv_paper.pub_url
            self.comment = arxiv_paper.comment
            self.journal_ref = arxiv_paper.journal_ref

            self.links = arxiv_paper.links
            self.authors = '#'.join([str(i) for i in arxiv_paper.authors])
            self.gpt_keywords = None
            self.gpt_keyword = None
            self.search_by_keywords = search_by_keywords
            self.is_review = 1
            self.valid = 1
            self.last_update_time = datetime.now()
            for key, value in arxiv_paper.__dict__.items():
                if not key.startswith('_'):
                    setattr(self, key, value)
        if s2_paper is not None:
            self.s2_id = s2_paper.s2id
            self.s2_publication_date = s2_paper.publication_date
            self.s2_tldr = s2_paper.tldr
            self.s2_pub_info = s2_paper.publisher#str(s2_paper.publisher) + '@' + str(s2_paper.publication_source)
            self.s2_pub_url = s2_paper.pub_url
            self.s2_citation_count = s2_paper.citation_count
            self.s2_reference_count = s2_paper.reference_count
            self.s2_field = s2_paper.field
            self.s2_influential_citation_count = s2_paper.influential_citation_count
            self.valid = 1
            self.authors_num = len(s2_paper.authors)
            self.last_update_time = datetime.now()




class CoP(Base):
    __tablename__ = 'cop'  # 
    idCoP = Column(String(36), primary_key=True)
    s2_id = Column(String(45),unique=True)
    citation = Column(Text)
    full_citation = Column(Text)


    def __init__(self, s2_id,citation):
        # 
        self.idCoP = str(uuid.uuid4())
        self.s2_id = s2_id
        self.citation = citation
        self.full_citation = None