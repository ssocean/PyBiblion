import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from retrievers.Author import Author
from retrievers.Publication import Document
import logging
from tools.llm_infer_util import get_chatgpt_field, get_chatgpt_fields
S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
from .metric_util import *
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import re
from datetime import datetime
from retry import retry
# OPENAI SETUP
import requests
from cfg.cache_manager import *
from cfg.config import *



class S2paper(Document):
    def __init__(self, ref_obj, ref_type='title', filled_authors=True, force_return=False, use_cache=True,**kwargs):
        '''

        :param ref_obj: search keywords
        :param ref_type: title   entity
        :param filled_authors:  retrieve detailed info about authors?
        :param force_return:  even title is not mapping, still return the result
        :param kwargs:
        '''
        super().__init__(ref_obj, **kwargs)
        self.ref_type = ref_type
        # Expectation: A typical program is unlikely to create more than 5 of these.
        self.S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
        self.S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.CACHE_FILE = ".ppicache"
        self.DEFAULT_TIMEOUT = 3.05  # 100 requests per 5 minutes
        self._entity = None
        # a few settings
        self.filled_authors = filled_authors
        self.force_return = force_return
        self._gpt_keyword = None
        self._gpt_keywords = None
        self._TNCSI = None
        self._TNCSI_S = None
        self._IEI = None
        self._RQM = None
        self._RUI = None
        self.use_cache = use_cache

    def __eq__(self, other):
        if isinstance(other, S2paper):
            return self.s2id == other.s2id
        return False

    def __hash__(self):
        return hash(self.s2id)
    @property
    @retry()
    def entity(self, max_tries=5):
        if self.ref_type == 'entity':
            self._entity = self.ref_obj
            return self._entity

        if self._entity is None:
            url = f"{self.S2_QUERY_URL}?query={self.ref_obj}&fieldsOfStudy=Computer Science&fields=url,title,abstract,authors,venue,externalIds,referenceCount,tldr,openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,publicationVenue&offset=0&limit=1"
            if url in disk_cache and self.use_cache:
                response = disk_cache[url]
            else:
                session = requests.Session()
                if s2api is not None:
                    headers = {
                        'x-api-key': s2api
                    }
                else:
                    headers = None
                reply = session.get(url, headers=headers)
                response = reply.json()
                disk_cache[url] = response

            if "data" not in response:
                self._entity = False
                return self._entity
            else:

                if self.ref_type == 'title' and re.sub(r'\W+', '', response['data'][0]['title'].lower()) != re.sub(
                        r'\W+', '', self.ref_obj.lower()):
                    if self.force_return:
                        self._entity = response['data'][0]
                        return self._entity
                    else:
                        print(response['data'][0]['title'].lower())
                        self._entity = False
                        return self._entity
                else:
                    self._entity = response['data'][0]
                    return self._entity
        return self._entity


    @property
    def title(self):
        if self.ref_type == 'title':
            return self.ref_obj.lower()
        if self.entity:
            return self.entity.get('title').lower()
        return None

    @property
    def publication_date(self):
        """The data of publication."""
        if self.entity:
            # if 'publicationDate' in self.entity and self.entity['publicationDate'] is not None:
            if self.entity.get('publicationDate') is not None:
                return datetime.strptime(self.entity['publicationDate'], "%Y-%m-%d")
        return None

    @property
    def s2id(self):
        """The `DocumentIdentifier` of this document."""
        return self.entity['paperId'] if self.entity else None

    @property
    def tldr(self):
        """The `DocumentIdentifier` of this document."""
        if self.entity:
            # if 'tldr' in self.entity and self.entity['tldr'] is not None:
            if self.entity.get('tldr') is not None:
                return self.entity['tldr']['text']
        return None

    @property
    def DOI(self):
        if self.entity:
            return self.entity.get('DOI')  # is not None:
        return None

    @property
    @retry()
    def authors(self):
        """The authors of this document."""
        if not self.entity:
            return None

        authors = []

        if 'authors' in self.entity:
            if not self.filled_authors:
                for item in self.entity['authors']:
                    author = Author(item['name'], _s2_id=item['authorId'])
                    authors.append(author)
                return authors
            else:
                url = (
                    f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/authors'
                    f'?fields=authorId,externalIds,name,affiliations,homepage,paperCount,'
                    f'citationCount,hIndex,url'
                )
                if url in disk_cache and self.use_cache:
                    response = disk_cache[url]
                else:
                    headers = {'x-api-key': s2api} if s2api else None
                    reply = requests.get(url, headers=headers)
                    response = reply.json()
                    disk_cache[url] = response

                for item in response.get('data', []):
                    author = Author(
                        item.get('name'),
                        _s2_id=item.get('authorId'),
                        _s2_url=item.get('url'),
                        _h_index=item.get('hIndex'),
                        _citationCount=item.get('citationCount'),
                        _paperCount=item.get('paperCount'),
                    )
                    authors.append(author)

                return authors

        return None

    @property
    def affiliations(self):
        if self.authors:
            affiliations = []
            for author in self.authors:
                if author.affiliations is not None:
                    affiliations.append(author.affiliations)
            return ';'.join(list(set(affiliations)))
        return None

    @property
    def publisher(self):
        """The publisher of this document."""
        if self.entity:
            return self.entity.get('publicationVenue')
        return None

    @property
    def publication_source(self):
        """The name of the publication source (i.e., journal name,
        conference name, etc.)
        """
        if self.entity:
            return self.entity.get('venue')
        return None

    @property
    def source_type(self):
        if self.entity:
            return self.entity.get('publicationTypes')
        return None

    @property
    def abstract(self):
        """The abstract of this document."""
        if self.entity:
            return self.entity.get('abstract')
        return None

    @property
    def pub_url(self):
        """The list of other documents that cite this document."""
        if self.entity:
            return self.entity.get('openAccessPdf')
        return None

    @property
    def citation_count(self):
        if self.entity:
            return self.entity.get('citationCount')
        return None

    @property
    def reference_count(self):
        if self.entity:
            return self.entity.get('referenceCount')
        return None

    @property
    def field(self):
        if self.entity:
            if self.entity.get('s2FieldsOfStudy') is not None:
                fields = []
                for fdict in self.entity.get('s2FieldsOfStudy'):
                    category = fdict['category']
                    fields.append(category)
                fields = ','.join(list(set(fields)))
                return fields
        return None

    @property
    def influential_citation_count(self):
        if self.entity:
            return self.entity.get('influentialCitationCount')
        return None

    @property
    @retry()
    def references(self):
        if self.entity:
            references = []
            url = f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/references?fields=authors,contexts,intents,isInfluential,venue,title,authors,citationCount,influentialCitationCount,publicationDate,venue&limit=999'

            if url in disk_cache and self.use_cache:
                response = disk_cache[url]
            else:
                session = requests.Session()
                headers = {'x-api-key': s2api} if s2api else None
                reply = session.get(url, headers=headers)
                response = reply.json()
                disk_cache[url] = response

            if 'data' not in response:
                return []

            for item in response['data']:
                ref = S2paper(item['citedPaper']['title'])
                ref.filled_authors = False
                info = {
                    'paperId': item['citedPaper']['paperId'],
                    'contexts': item['contexts'],
                    'intents': item['intents'],
                    'isInfluential': item['isInfluential'],
                    'title': item['citedPaper']['title'],
                    'venue': item['citedPaper']['venue'],
                    'citationCount': item['citedPaper']['citationCount'],
                    'influentialCitationCount': item['citedPaper']['influentialCitationCount'],
                    'publicationDate': item['citedPaper']['publicationDate'],
                }
                ref._entity = info
                references.append(ref)
            return references

        return None

    @property
    @retry()
    def citations_detail(self):
        if not self.entity:
            return None

        results = []
        offset = 0
        session = requests.Session()
        headers = {'x-api-key': s2api} if s2api else None

        while True:
            url = (
                f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/citations'
                f'?fields=authors,contexts,intents,isInfluential,venue,title,authors,'
                f'citationCount,influentialCitationCount,publicationDate,venue'
                f'&limit=1000&offset={offset}'
            )
            offset += 1000

            # 使用 diskcache 缓存
            if url in disk_cache and self.use_cache:
                response = disk_cache[url]
            else:
                reply = session.get(url, headers=headers)
                response = reply.json()
                disk_cache[url] = response

            data = response.get('data', [])
            if not data:
                break

            for item in data:
                citing_paper = item.get('citingPaper', {})
                ref = S2paper(citing_paper.get('title', ''))
                ref.filled_authors = True

                ref._entity = {
                    'paperId': citing_paper.get('paperId'),
                    'contexts': item.get('contexts'),
                    'intents': item.get('intents'),
                    'isInfluential': item.get('isInfluential'),
                    'title': citing_paper.get('title'),
                    'venue': citing_paper.get('venue'),
                    'citationCount': citing_paper.get('citationCount'),
                    'influentialCitationCount': citing_paper.get('influentialCitationCount'),
                    'publicationDate': citing_paper.get('publicationDate'),
                    'authors': citing_paper.get('authors'),
                }

                results.append(ref)

        return results
    @property
    def gpt_keyword(self):
        # only one keyword is generated
        if self._gpt_keyword is None:
            self._gpt_keyword = get_chatgpt_field(self.title, self.abstract, extra_prompt=True)
        return self._gpt_keyword

    @property
    def gpt_keywords(self):
        # get multiple keyword at one time
        if self._gpt_keywords is None:
            self._gpt_keywords = get_chatgpt_fields(self.title, self.abstract, extra_prompt=True)
        return self._gpt_keywords
    @property
    @retry(tries=5)
    def TNCSI(self):
        if self._TNCSI is None:
            self._TNCSI = get_TNCSI(self,show_PDF=False)
        return self._TNCSI

    @property
    @retry(tries=5)
    def TNCSI_S(self):
        if self._TNCSI_S is None:
            kwd = self.gpt_keyword
            self._TNCSI_S = get_TNCSI(self,topic_keyword=kwd,show_PDF=False,same_year=True)
        return self._TNCSI_S

    @property
    @retry(tries=5)
    def IEI(self):
        if self.publication_date is not None and self.citation_count != 0:
            if self._IEI is None:
                self._IEI = get_IEI(self.title, normalized=False, exclude_last_n_month=1,show_img=False)
            return self._IEI
        rst = {}
        rst['L6'] = float('-inf')
        rst['I6'] = float('-inf')
        return rst

    @property
    @retry(tries=5)
    def RQM(self):
        if self.publication_date is not None and self.reference_count != 0:
            if self._RQM is None:
                self._RQM = get_RQM(self, tncsi_rst=self.TNCSI, beta=5)
            return self._RQM
        return {}
    @property
    @retry(tries=5)
    def RUI(self):
        if self.publication_date is not None and self.reference_count != 0:
            if self._RUI is None:
                self._RUI = get_RUI(self)
            return self._RUI
        return {}

# sp = S2paper('segment anything')

# print(sp.references)
# print(sp.influential_citation_count)
# S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
# S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
#
#
# def request_query(query, sort_rule=None, pub_date: datetime = None, continue_token=None):
#     '''
#     :param query: keyword
#     :param sort_rule: publicationDate:asc - return oldest papers first. || citationCount:desc - return most highly-cited papers first. ||paperId - return papers in ID order, low-to-high.
#     :param pub_date:
#     2019-03-05 on March 3rd, 2019
#     2019-03 during March 2019
#     2019 during 2019
#     2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
#     1981-08-25: on or after August 25th, 1981
#     :2015-01 before or on January 31st, 2015
#     2015:2020 between January 1st, 2015 and December 31st, 2020
#     :return:
#     '''
#     s2api = None  # 全局变量也可以提前放外面
#     p_dict = dict(query=query)
#
#     if pub_date:
#         p_dict['publicationDateOrYear'] = f':{pub_date.year}-{pub_date.month}-{pub_date.day}'
#     if continue_token:
#         p_dict['token'] = continue_token
#     if sort_rule:
#         p_dict['sort'] = sort_rule
#
#     params = urlencode(p_dict)
#     url = (
#         f"{S2_QUERY_URL}?{params}&fields=url,title,abstract,authors,venue,externalIds,referenceCount,"
#         f"openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,"
#         f"s2FieldsOfStudy,publicationTypes,publicationDate"
#     )
#
#     # 使用 diskcache 缓存
#     if url in disk_cache:
#         response = disk_cache[url]
#     else:
#         session = requests.Session()
#         headers = {'x-api-key': s2api} if s2api else None
#         reply = session.get(url, headers=headers)
#         response = reply.json()
#         disk_cache[url] = response
#
#     if "data" not in response:
#         msg = response.get("error") or response.get("message") or "unknown"
#         raise Exception(f"error while fetching {url}: {msg}")
#
#     return response
#
# def relevance_query(query, pub_date: datetime = None):
#     '''
#     :param query: keyword
#     :param sort_rule: publicationDate:asc - return oldest papers first. || citationCount:desc - return most highly-cited papers first. ||paperId - return papers in ID order, low-to-high.
#     :param pub_date:
#     2019-03-05 on March 3rd, 2019
#     2019-03 during March 2019
#     2019 during 2019
#     2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
#     1981-08-25: on or after August 25th, 1981
#     :2015-01 before or on January 31st, 2015
#     2015:2020 between January 1st, 2015 and December 31st, 2020
#     :return:
#     '''
#     s2api = None
#     p_dict = dict(query=query)
#
#     if pub_date:
#         p_dict['publicationDateOrYear'] = f':{pub_date.year}-{pub_date.month}-{pub_date.day}'
#
#     params = urlencode(p_dict)
#     url = (
#         f"https://api.semanticscholar.org/graph/v1/paper/search?{params}&fields="
#         f"url,title,abstract,authors,venue,referenceCount,openAccessPdf,"
#         f"citationCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,"
#         f"publicationTypes,publicationDate"
#     )
#
#     if url in disk_cache:
#         response = disk_cache[url]
#     else:
#         session = requests.Session()
#         headers = {'x-api-key': s2api} if s2api else None
#         reply = session.get(url, headers=headers)
#         response = reply.json()
#         disk_cache[url] = response
#
#     if "data" not in response:
#         msg = response.get("error") or response.get("message") or "unknown"
#         raise Exception(f"error while fetching {url}: {msg}")
#
#     return response
#
#
#
# class S2paper(Document):
#     def __init__(self, ref_obj, ref_type='title', filled_authors=True, force_return=False, use_cache=True, **kwargs):
#         '''
#
#         :param ref_obj: search keywords
#         :param ref_type: title   entity
#         :param filled_authors:  retrieve detailed info about authors?
#         :param force_return:  even title is not mapping, still return the result
#         :param kwargs:
#         '''
#         super().__init__(ref_obj, **kwargs)
#         self.ref_type = ref_type
#         # Expectation: A typical program is unlikely to create more than 5 of these.
#         self.S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
#         self.S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
#         self.CACHE_FILE = ".ppicache"
#         self.DEFAULT_TIMEOUT = 3.05  # 100 requests per 5 minutes
#         self._entity = None
#         # a few settings
#         self.filled_authors = filled_authors
#         self.force_return = force_return
#         self._gpt_keyword = None
#         self._gpt_keywords = None
#         self._TNCSI = None
#         self._TNCSI_S = None
#         self._IEI = None
#         self._RQM = None
#         self._RUI = None
#         self.headers = None
#         if s2api is not None:
#             self.headers = {
#                 'x-api-key': s2api
#             }
#
#
#         self.use_cache = use_cache
#
#
#
#     # @retry()
#     @property
#     def entity(self, max_tries=5):
#         if self.ref_type == 'entity':
#             self._entity = self.ref_obj
#             return self._entity
#
#         if self._entity is None:
#             url = f"{self.S2_QUERY_URL}?query={self.ref_obj}&fieldsOfStudy=Computer Science&fields=url,title,abstract,authors,venue,externalIds,referenceCount,tldr,openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,publicationVenue&offset=0&limit=1"
#             reply = cached_get(url, headers=self.headers)
#             response = reply.json()
#             if "data" not in response:
#                 msg = response.get("error") or response.get("message") or "unknown"
#                 self._entity = False
#                 return self._entity
#                 # raise Exception(f"error while fetching {reply.url}: {msg}")
#             else:
#
#                 if self.ref_type == 'title' and re.sub(r'\W+', '', response['data'][0]['title'].lower()) != re.sub(
#                         r'\W+', '', self.ref_obj.lower()):
#                     if self.force_return:
#                         self._entity = response['data'][0]
#                         return self._entity
#                     else:
#                         print(response['data'][0]['title'].lower())
#                         self._entity = False
#                         return self._entity
#                 else:
#                     self._entity = response['data'][0]
#                     return self._entity
#         return self._entity
#
#     @property
#     def gpt_keyword(self):
#         # only one keyword is generated
#         if self._gpt_keyword is None:
#             self._gpt_keyword = get_chatgpt_field(self.title, self.abstract, extra_prompt=True)
#         return self._gpt_keyword
#
#     @property
#     def gpt_keywords(self):
#         # get multiple keyword at one time
#         if self._gpt_keywords is None:
#             self._gpt_keywords = get_chatgpt_fields(self.title, self.abstract, extra_prompt=True)
#         return self._gpt_keywords
#
#     @property
#     def title(self):
#         if self.ref_type == 'title':
#             return self.ref_obj.lower()
#         if self.entity:
#             return self.entity.get('title').lower()
#         return None
#
#     @property
#     def publication_date(self):
#         """The data of publication."""
#         if self.entity:
#             # if 'publicationDate' in self.entity and self.entity['publicationDate'] is not None:
#             if self.entity.get('publicationDate') is not None:
#                 return datetime.strptime(self.entity['publicationDate'], "%Y-%m-%d")
#         return None
#
#     @property
#     def s2id(self):
#         """The `DocumentIdentifier` of this document."""
#         return self.entity['paperId'] if self.entity else None
#
#     @property
#     def tldr(self):
#         """The `DocumentIdentifier` of this document."""
#         if self.entity:
#             # if 'tldr' in self.entity and self.entity['tldr'] is not None:
#             if self.entity.get('tldr') is not None:
#                 return self.entity['tldr']['text']
#         return None
#
#     @property
#     def DOI(self):
#         if self.entity:
#             # if 'DOI' in self.entity['externalIds'] and self.entity['externalIds']['DOI'] is not None:
#             return self.entity.get('DOI')  # is not None:
#             # return self.entity['externalIds']['DOI']
#         return None
#
#     @property
#     @retry()
#     def authors(self):
#         """The authors of this document."""
#         if self.entity:
#             authors = []
#             if 'authors' in self.entity:
#                 if not self.filled_authors:
#                     for item in self.entity['authors']:
#                         author = Author(item['name'], _s2_id=item['authorId'])
#                         # author.entity
#                         authors.append(author)
#                     return authors
#                 else:
#                     url = (f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/authors?fields=authorId,'
#                            f'externalIds,name,affiliations,homepage,paperCount,citationCount,hIndex,url')
#                     reply = cached_get(url, headers=self.headers)
#
#                     for item in reply.json()['data']:
#                         author = Author(item['name'], _s2_id=item['authorId'], _s2_url=item['url'],
#                                         _h_index=item['hIndex'], _citationCount=item['citationCount'],
#                                         _paperCount=item['paperCount'])
#                         authors.append(author)
#                     return authors
#
#         return None
#
#     @property
#     def affiliations(self):
#         if self.authors:
#             affiliations = []
#             for author in self.authors:
#                 if author.affiliations is not None:
#                     affiliations.append(author.affiliations)
#             return ';'.join(list(set(affiliations)))
#         return None
#
#     @property
#     def publisher(self):
#         """The publisher of this document."""
#         if self.entity:
#             return self.entity.get('publicationVenue')
#             # if 'publicationVenue' in self.entity and self.entity['publicationVenue'] is not None:
#             #     return self.entity['publicationVenue']
#         return None
#
#     @property
#     def publication_source(self):
#         """The name of the publication source (i.e., journal name,
#         conference name, etc.)
#         """
#         if self.entity:
#             # if 'venue' in self.entity and self.entity['venue'] is not None:
#             #     return self.entity['venue']
#             return self.entity.get('venue')
#         return None
#
#     @property
#     def source_type(self):
#         if self.entity:
#             # if 'publicationTypes' in self.entity and self.entity['publicationTypes'] is not None:
#             #     return self.entity['publicationTypes']
#             return self.entity.get('publicationTypes')
#         return None
#
#     @property
#     def abstract(self):
#         """The abstract of this document."""
#         if self.entity:
#             # if 'abstract' in self.entity and self.entity['abstract'] is not None:
#             #     return self.entity['abstract']
#             return self.entity.get('abstract')
#         return None
#
#     @property
#     def pub_url(self):
#         """The list of other documents that cite this document."""
#         if self.entity:
#             # if 'openAccessPdf' in self.entity:
#             #     return self.entity['openAccessPdf']
#             return self.entity.get('openAccessPdf')
#         return None
#
#     @property
#     def citation_count(self):
#
#         if self.entity:
#             # if 'citationCount' in self.entity:
#             #     return self.entity['citationCount']
#             return self.entity.get('citationCount')
#         return None
#
#     @property
#     def reference_count(self):
#
#         if self.entity:
#             # if 'citationCount' in self.entity:
#             #     return self.entity['referenceCount']
#             return self.entity.get('referenceCount')
#         return None
#
#     @property
#     def field(self):
#         if self.entity:
#             if self.entity.get('s2FieldsOfStudy') is not None:
#                 fields = []
#                 for fdict in self.entity.get('s2FieldsOfStudy'):
#                     category = fdict['category']
#                     fields.append(category)
#                 fields = ','.join(list(set(fields)))
#                 return fields
#         return None
#
#     @property
#     def influential_citation_count(self):
#         if self.entity:
#             # if 'influentialCitationCount' in self.entity:
#             #     return self.entity['influentialCitationCount']
#             return self.entity.get('influentialCitationCount')
#         return None
#
#     @property
#     @retry(delay=6)
#     def references(self):
#         if self.entity:
#             references = []
#             url = f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/references?fields=authors,contexts,intents,isInfluential,venue,title,authors,citationCount,influentialCitationCount,publicationDate,venue&limit=999'
#             reply = cached_get(url, headers=self.headers)
#
#
#             if 'data' not in reply.json() or reply is None or reply.json()['data'] is None or reply.json() is None:
#                 return []
#
#             for item in reply.json()['data']:
#                 # print(item)
#                 ref = S2paper(item['citedPaper']['title'])
#                 ref.filled_authors = False
#                 info = {'paperId': item['citedPaper']['paperId'], 'contexts': item['contexts'],
#                         'intents': item['intents'], 'isInfluential': item['isInfluential'],
#                         'title': item['citedPaper']['title'], 'venue': item['citedPaper']['venue'],
#                         'citationCount': item['citedPaper']['citationCount'],
#                         'influentialCitationCount': item['citedPaper']['influentialCitationCount'],
#                         'publicationDate': item['citedPaper']['publicationDate']}
#                 # authors = []
#
#                 ref._entity = info
#                 # print(ref.citation_count)
#                 references.append(ref)
#
#             return references
#
#
#         return None
#
#     @property
#     @retry()
#     def citations_detail(self):
#         if self.entity:
#             references = []
#             is_continue = True
#             offset = 0
#             while is_continue:
#
#                 url = f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/citations?fields=authors,contexts,intents,isInfluential,venue,title,authors,citationCount,influentialCitationCount,publicationDate,venue&limit=1000&offset={offset}'
#                 offset += 1000
#                 reply = cached_get(url, headers=self.headers)
#                 if 'data' not in reply.json() or reply.json()['data'] == []:
#                     is_continue = False
#                 for item in reply.json()['data']:
#                     # print(item)
#                     ref = S2paper(item['citingPaper']['title'])
#
#                     ref.filled_authors = True
#
#                     info = {'paperId': item['citingPaper']['paperId'], 'contexts': item['contexts'],
#                             'intents': item['intents'], 'isInfluential': item['isInfluential'],
#                             'title': item['citingPaper']['title'], 'venue': item['citingPaper']['venue'],
#                             'citationCount': item['citingPaper']['citationCount'],
#                             'influentialCitationCount': item['citingPaper']['influentialCitationCount'],
#                             'publicationDate': item['citingPaper']['publicationDate'],
#                             'authors': item['citingPaper']['authors']}
#                     # authors = []
#
#                     ref._entity = info
#                     # print(ref.citation_count)
#                     references.append(ref)
#             return references
#
#         return None
#     @property
#     @retry(tries=5)
#     def TNCSI(self):
#         if self._TNCSI is None:
#             self._TNCSI = get_TNCSI(self,show_PDF=False)
#         return self._TNCSI
#
#     @property
#     @retry(tries=5)
#     def TNCSI_S(self):
#         if self._TNCSI_S is None:
#             kwd = self.gpt_keyword
#             self._TNCSI_S = get_TNCSI(self,topic_keyword=kwd,show_PDF=False,same_year=True)
#         return self._TNCSI_S
#
#     @property
#     @retry(tries=5)
#     def IEI(self):
#         if self.publication_date is not None and self.citation_count != 0:
#             if self._IEI is None:
#                 self._IEI = get_IEI(self.title, normalized=False, exclude_last_n_month=1,show_img=False)
#             return self._IEI
#         rst = {}
#         rst['L6'] = float('-inf')
#         rst['I6'] = float('-inf')
#         return rst
#
#     @property
#     @retry(tries=5)
#     def RQM(self):
#         if self.publication_date is not None and self.reference_count != 0:
#             if self._RQM is None:
#                 self._RQM = get_RQM(self, tncsi_rst=self.TNCSI, beta=5)
#             return self._RQM
#         return {}
#     @property
#     @retry(tries=5)
#     def RUI(self):
#         if self.publication_date is not None and self.reference_count != 0:
#             if self._RUI is None:
#                 self._RUI = get_RUI(self)
#             return self._RUI
#         return {}
#
