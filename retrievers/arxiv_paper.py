
import warnings
import arxiv
from retrievers.Author import Author
from retrievers.Publication import Document


import re

def filter_punctuation(text: str) -> str:
    """
    Remove punctuation and special characters from the input string,
    retaining only alphanumerics and spaces.

    Args:
        text (str): Input string (e.g., a title)

    Returns:
        str: Cleaned string without punctuation
    """
    return re.sub(r'[^\w\s]', '', text)


def get_arxiv_id_from_url(url:str):
    if url.endswith('.pdf'):
        url = url[:-4]
    id = url.split(r'/')[-1]
    return id




class Arxiv_paper(Document):
    def __init__(self,ref_obj, ref_type='title', **kwargs):
        super().__init__(ref_obj, **kwargs)
        self.ref_type = ref_type
        # self._entity = None

    # @retry()
    @property
    def entity(self,max_tries=5):
        if self._entity is None:
            if self.ref_type == 'title':
                search = arxiv.Search(query=f'ti:{filter_punctuation(self.ref_obj)}')
                for i,matched_paper in enumerate(search.results()):
                    if matched_paper.title.lower().replace(" ", "") == self.ref_obj.lower().replace(" ", ""):
                        self._entity = matched_paper
                        return self._entity
                    if i>=max_tries:
                        warnings.warn(
                            "Haven't fetch anything from arxiv. ",
                            UserWarning)
                        self._entity = False
                        return self._entity
            elif self.ref_type == 'id':
                # print('searching id')
                search = arxiv.Search(id_list=[self.ref_obj])
                # print('searching done')
                matched_paper = next(search.results())
                # print('fetching Done')
                self._entity = matched_paper
                return self._entity
            elif self.ref_type == 'entity':
                self._entity = self.ref_obj
                return self._entity
        return self._entity


    @property
    def title(self):
        if self.ref_type == 'title':
            return self.ref_obj.lower()
        if self.entity:
            return self.entity.title.lower()
        return None
    @property
    def publication_date(self):
        """The data of publication."""
        return self.entity.published if self.entity.published else None

    @property
    def id(self):
        """The `DocumentIdentifier` of this document."""
        return self.entity.entry_id if self.entity.entry_id else None

    @property
    def authors(self):
        """The authors of this document."""
        if self.entity:
            return [Author(i.name) for i in self.entity.authors]
        return None



    @property
    def publisher(self):
        """The publisher of this document."""
        return 'arxiv'



    @property
    def publication_source(self):
        """The name of the publication source (i.e., journal name,
        conference name, etc.)
        """
        return 'arxiv'

    @property
    def source_type(self):
        """The type of publication source (i.e., journal, conference
        proceedings, book, etc.)
        """
        return 'pre-print'


    @property
    def abstract(self):
        """The abstract of this document."""
        return self.entity.summary if self.entity.summary else None



    @property
    def pub_url(self):
        """The list of other documents that cite this document."""
        return self.entity.entry_id if self.entity.entry_id else None

    @property
    def comment(self):
        '''The authors comment if present. '''
        return self.entity.comment if self.entity.comment else None
    @property
    def journal_ref(self):
        '''A journal reference if present. '''
        return self.entity.journal_ref if self.entity.journal_ref else None
    @property
    def primary_category(self):
        '''The primary arXiv category. '''
        return self.entity.primary_category if self.entity.primary_category else None
    @property
    def categories(self):
        '''The arXiv or ACM or MSC category for an article if present.'''
        return self.entity.categories if self.entity.categories else None
    @property
    def links(self):
        '''Can be up to 3 given url's associated with this article. '''
        return self.entity.links if self.entity.links else None

