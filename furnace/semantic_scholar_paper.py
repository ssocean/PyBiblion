import re
import shelve
from urllib.parse import urlencode
from datetime import datetime
from retry import retry
# OPENAI SETUP
import requests
from CACHE.CACHE_Config import generate_cache_file_name
from config.config import *
from furnace.Author import Author
from furnace.Publication import Document
from tools.gpt_util import get_chatgpt_field, get_chatgpt_fields


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
        self.use_cache = use_cache


    @property
    @retry()
    def entity(self, max_tries=5):
        if self.ref_type == 'entity':
            self._entity = self.ref_obj
            return self._entity

        if self._entity is None:
            url = f"{self.S2_QUERY_URL}?query={self.ref_obj}&fieldsOfStudy=Computer Science&fields=url,title,abstract,authors,venue,externalIds,referenceCount,tldr,openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,publicationDate,publicationVenue&offset=0&limit=1"
            with shelve.open(generate_cache_file_name(url)) as cache:

                # params = urlencode(dict(query=self.ref_obj, offset=0, limit=1))
                # params = quote_plus(self.ref_obj)
                # print(params)
                # fieldsOfStudy=Computer Science&
                if url in cache and self.use_cache:
                    reply = cache[url]
                else:
                    session = requests.Session()
                    if s2api is not None:
                        headers = {
                            'x-api-key': s2api
                        }
                    else:
                        headers = None
                    reply = session.get(url, headers=headers)
                    cache[url] = reply

            response = reply.json()
            if "data" not in response:
                msg = response.get("error") or response.get("message") or "unknown"
                self._entity = False
                return self._entity
                # raise Exception(f"error while fetching {reply.url}: {msg}")
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
    def gpt_keyword(self):
        # only one keyword is generated
        if self._gpt_keyword is None:
            self._gpt_keyword = get_chatgpt_field(self.title,self.abstract,extra_prompt=True)
        return self._gpt_keyword

    @property
    def gpt_keywords(self):
        # get multiple keyword at one time
        if self._gpt_keywords is None:
            self._gpt_keywords = get_chatgpt_fields(self.title,self.abstract,extra_prompt=True)
        return self._gpt_keywords

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
            # if 'DOI' in self.entity['externalIds'] and self.entity['externalIds']['DOI'] is not None:
            return self.entity.get('DOI')  # is not None:
            # return self.entity['externalIds']['DOI']
        return None

    @property
    @retry()
    def authors(self):
        """The authors of this document."""
        if self.entity:
            authors = []
            if 'authors' in self.entity:
                if not self.filled_authors:
                    for item in self.entity['authors']:
                        author = Author(item['name'], _s2_id=item['authorId'])
                        # author.entity
                        authors.append(author)
                    return authors
                else:
                    if s2api is not None:
                        headers = {
                            'x-api-key': s2api
                        }
                    else:
                        headers = None
                    r = requests.get(
                        f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/authors?fields=authorId,externalIds,name,affiliations,homepage,paperCount,citationCount,hIndex,url',
                        headers=headers
                    )
                    for item in r.json()['data']:
                        author = Author(item['name'], _s2_id=item['authorId'], _s2_url=item['url'],
                                        _h_index=item['hIndex'], _citationCount=item['citationCount'],
                                        _paperCount=item['paperCount'])
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
            # if 'publicationVenue' in self.entity and self.entity['publicationVenue'] is not None:
            #     return self.entity['publicationVenue']
        return None

    @property
    def publication_source(self):
        """The name of the publication source (i.e., journal name,
        conference name, etc.)
        """
        if self.entity:
            # if 'venue' in self.entity and self.entity['venue'] is not None:
            #     return self.entity['venue']
            return self.entity.get('venue')
        return None

    @property
    def source_type(self):
        if self.entity:
            # if 'publicationTypes' in self.entity and self.entity['publicationTypes'] is not None:
            #     return self.entity['publicationTypes']
            return self.entity.get('publicationTypes')
        return None

    @property
    def abstract(self):
        """The abstract of this document."""
        if self.entity:
            # if 'abstract' in self.entity and self.entity['abstract'] is not None:
            #     return self.entity['abstract']
            return self.entity.get('abstract')
        return None

    @property
    def pub_url(self):
        """The list of other documents that cite this document."""
        if self.entity:
            # if 'openAccessPdf' in self.entity:
            #     return self.entity['openAccessPdf']
            return self.entity.get('openAccessPdf')
        return None

    @property
    def citation_count(self):

        if self.entity:
            # if 'citationCount' in self.entity:
            #     return self.entity['citationCount']
            return self.entity.get('citationCount')
        return None

    @property
    def reference_count(self):

        if self.entity:
            # if 'citationCount' in self.entity:
            #     return self.entity['referenceCount']
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
            # if 'influentialCitationCount' in self.entity:
            #     return self.entity['influentialCitationCount']
            return self.entity.get('influentialCitationCount')
        return None

    def plot_s2citaions(keyword: str, year: str = '2018-2023', total_num=100, CACHE_FILE='.semantischolar'):
        l = 0
        citation_count = []
        influentionCC = []

        return citation_count, influentionCC

    @property
    @retry()
    def references(self):
        if self.entity:
            references = []
            url = f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/references?fields=authors,contexts,intents,isInfluential,venue,title,authors,citationCount,influentialCitationCount,publicationDate,venue&limit=999'

            with shelve.open(generate_cache_file_name(url)) as cache:
                # print(url) #references->citations
                if url in cache:
                    r = cache[url]
                else:
                    if s2api is not None:
                        headers = {
                            'x-api-key': s2api
                        }
                    else:
                        headers = None
                    r = requests.get(url, headers=headers)
                    cache[url] = r
                if 'data' not in r.json():
                    return []
                for item in r.json()['data']:
                    # print(item)
                    ref = S2paper(item['citedPaper']['title'])
                    ref.filled_authors = False
                    info = {'paperId': item['citedPaper']['paperId'], 'contexts': item['contexts'],
                            'intents': item['intents'], 'isInfluential': item['isInfluential'],
                            'title': item['citedPaper']['title'], 'venue': item['citedPaper']['venue'],
                            'citationCount': item['citedPaper']['citationCount'],
                            'influentialCitationCount': item['citedPaper']['influentialCitationCount'],
                            'publicationDate': item['citedPaper']['publicationDate']}
                    # authors = []

                    ref._entity = info
                    # print(ref.citation_count)
                    references.append(ref)
                return references

        return None

    @property
    @retry()
    def citations_detail(self):
        if self.entity:
            references = []
            is_continue = True
            offset = 0
            while is_continue:

                url = f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/citations?fields=authors,contexts,intents,isInfluential,venue,title,authors,citationCount,influentialCitationCount,publicationDate,venue&limit=1000&offset={offset}'
                offset += 1000
                with shelve.open(generate_cache_file_name(url)) as cache:
                    # print(url) #references->citations
                    if url in cache:
                        r = cache[url]
                    else:
                        if s2api is not None:
                            headers = {
                                'x-api-key': s2api
                            }
                        else:
                            headers = None
                        r = requests.get(url, headers=headers)
                        cache[url] = r
                    if 'data' not in r.json() or r.json()['data'] == []:
                        is_continue=False
                    for item in r.json()['data']:
                        # print(item)
                        ref = S2paper(item['citingPaper']['title'])

                        ref.filled_authors = True

                        info = {'paperId': item['citingPaper']['paperId'], 'contexts': item['contexts'],
                                'intents': item['intents'], 'isInfluential': item['isInfluential'],
                                'title': item['citingPaper']['title'], 'venue': item['citingPaper']['venue'],
                                'citationCount': item['citingPaper']['citationCount'],
                                'influentialCitationCount': item['citingPaper']['influentialCitationCount'],
                                'publicationDate': item['citingPaper']['publicationDate'],'authors':item['citingPaper']['authors']}
                        # authors = []

                        ref._entity = info
                        # print(ref.citation_count)
                        references.append(ref)
            return references

        return None
# sp = S2paper('segment anything')

# print(sp.references)
# print(sp.influential_citation_count)
S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
CACHE_FILE = r"C:\Users\Ocean\Documents\GitHub\Dynamic_Literature_Review\CACHE\.queryCache"

def request_query(query, sort_rule=None, pub_date: datetime = None, continue_token=None):
    '''
    :param query: keyword
    :param sort_rule: publicationDate:asc - return oldest papers first. || citationCount:desc - return most highly-cited papers first. ||paperId - return papers in ID order, low-to-high.
    :param pub_date:
    2019-03-05 on March 3rd, 2019
    2019-03 during March 2019
    2019 during 2019
    2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
    1981-08-25: on or after August 25th, 1981
    :2015-01 before or on January 31st, 2015
    2015:2020 between January 1st, 2015 and December 31st, 2020
    :return:
    '''
    s2api = None
    p_dict = dict(query=query)
    if pub_date:
        p_dict['publicationDateOrYear'] = f':{pub_date.year}-{pub_date.month}-{pub_date.day}'
    if continue_token:
        p_dict['token'] = continue_token
    if sort_rule:
        p_dict['sort'] = sort_rule
    params = urlencode(p_dict)
    url = (f"{S2_QUERY_URL}?{params}&fields=url,title,abstract,authors,venue,externalIds,referenceCount,"
           f"openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,"
           f"s2FieldsOfStudy,publicationTypes,publicationDate")
    with shelve.open(generate_cache_file_name(url)) as cache:

        # if pub_date:
        #     url = url+f'$publicationDateOrYear=:{pub_date.year}-{pub_date.month}-{pub_date.day}'
        # if continue_token:
        #     url = url+f'$token={continue_token}'
        # print(url)
        if url in cache:
            reply = cache[url]
        else:
            session = requests.Session()
            if s2api is not None:
                headers = {
                    'x-api-key': s2api
                }
            else:
                headers = None
            reply = session.get(url, headers=headers)
            cache[url] = reply

            reply = session.get(url)
        response = reply.json()

        if "data" not in response:
            msg = response.get("error") or response.get("message") or "unknown"
            raise Exception(f"error while fetching {reply.url}: {msg}")

        return response

def relevance_query(query, pub_date: datetime = None):
    '''
    :param query: keyword
    :param sort_rule: publicationDate:asc - return oldest papers first. || citationCount:desc - return most highly-cited papers first. ||paperId - return papers in ID order, low-to-high.
    :param pub_date:
    2019-03-05 on March 3rd, 2019
    2019-03 during March 2019
    2019 during 2019
    2016-03-05:2020-06-06 as early as March 5th, 2016 or as late as June 6th, 2020
    1981-08-25: on or after August 25th, 1981
    :2015-01 before or on January 31st, 2015
    2015:2020 between January 1st, 2015 and December 31st, 2020
    :return:
    '''
    s2api = None
    p_dict = dict(query=query)
    if pub_date:
        p_dict['publicationDateOrYear'] = f':{pub_date.year}-{pub_date.month}-{pub_date.day}'

    params = urlencode(p_dict)
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search?{params}&fields=url,title,abstract,authors,venue,referenceCount,"
           f"openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,"
           f"s2FieldsOfStudy,publicationTypes,publicationDate")
    with shelve.open(generate_cache_file_name(url)) as cache:

        # if pub_date:
        #     url = url+f'$publicationDateOrYear=:{pub_date.year}-{pub_date.month}-{pub_date.day}'
        # if continue_token:
        #     url = url+f'$token={continue_token}'
        # print(url)
        if url in cache:
            reply = cache[url]
        else:
            session = requests.Session()
            if s2api is not None:
                headers = {
                    'x-api-key': s2api
                }
            else:
                headers = None
            reply = session.get(url, headers=headers)
            cache[url] = reply

            reply = session.get(url)
        response = reply.json()

        if "data" not in response:
            msg = response.get("error") or response.get("message") or "unknown"
            raise Exception(f"error while fetching {reply.url}: {msg}")

        return response
