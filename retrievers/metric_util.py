import random
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from retry import retry
from tqdm import tqdm
import math
import statistics
from collections import OrderedDict
from datetime import datetime, timedelta
from cfg.config import s2api
S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
CACHE_FILE = r"CACHE\.queryCache"
from filelock import Timeout, FileLock
from CACHE.cache_request import generate_cache_file_name, cached_get
import requests
from urllib.parse import urlencode
import shelve
from scipy.integrate import cumtrapz
from scipy import stats
from retry import retry
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb



def _get_TNCSI_score(citation:int, loc, scale):
    import math
    def exponential_cdf(x, loc, scale):
        if x < loc:
            return 0
        else:
            z = (x - loc) / scale
            cdf = 1 - math.exp(-z)
            return cdf

    # print(citation, loc, scale)
    TNCSI = exponential_cdf(citation, loc, scale)
    return TNCSI
def get_PDF_curve_params(topic, topk=2000, show_img=False, pub_date:datetime=None, mode=1):

    citation, _ = get_citation_discrete_distribution(topic, total_num=topk, pub_date=pub_date, mode=mode)
    citation = np.array(citation)

    # 
    try:
        params = stats.expon.fit(citation)
    except:
        # print(citation)
        return None,None
    loc, scale = params
    if len(citation)<=1:
        return None, None

    x = np.linspace(np.min(citation), np.max(citation), 100)
    pdf = stats.expon.pdf(x, loc, scale)

    if show_img:
        plt.clf()
        plt.figure(figsize=(6, 4))
        plt.hist(citation, bins=1000, density=True, alpha=0.5)
        plt.plot(x, pdf, 'r', label='Fitted Exponential Distribution')
        plt.xlabel('Number of Citations')
        plt.ylabel('Frequency')
        plt.legend()
        # print('saving success')
        # plt.savefig(save_img_pth)
        plt.show()

    return loc, scale  # , image
@retry(delay=6)
def get_citation_discrete_distribution(keyword: str, total_num=1000, pub_date:datetime=None, mode=1):
    '''

    '''
    citation_count = []
    influentionCC = []
    date_six_months_ago = None
    date_six_months_later = None
    publicationDateOrYear = ''
    if pub_date:
        six_months = timedelta(days=183)  #
        date_six_months_ago = pub_date - six_months
        date_six_months_later = pub_date + six_months
        publicationDateOrYear = f"&publicationDateOrYear={date_six_months_ago.strftime('%Y-%m')}:{date_six_months_later.strftime('%Y-%m')}"


    if mode == 1: # default search
        # publicationDateOrYear 2016-03-05:2020-06-06
        for i in range(int(total_num // 100)):
            url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}{publicationDateOrYear}&fieldsOfStudy=Computer Science&fields=title,year,citationCount,influentialCitationCount&offset={100 * i}&limit=100'
            # print(url)
            headers = None
            if s2api is not None:
                headers = {
                    'x-api-key': s2api
                }

            r = cached_get(url,headers=headers)

            try:
                response = r.json()
            except Exception as e:
                return citation_count, influentionCC
            if 'data' not in response:
                # logger.info(f'Fetching {l} data from SemanticScholar.')
                return citation_count, influentionCC
                # raise ConnectionError
            for item in r.json()['data']:
                if int(item['citationCount']) >= 0:
                    citation_count.append(int(item['citationCount']))
                    influentionCC.append(int(item['influentialCitationCount']))
                else:
                    print(item['citationCount'])

        return citation_count, influentionCC

    elif mode == 2: # bulk search
        query = f'"{keyword}"~3'
        continue_token = None
        C=1000
        for i in range(0, C, 1000):
            if continue_token is None:
                response = request_query(query, early_date=date_six_months_ago,later_date=date_six_months_later)
            else:
                response = request_query(query,continue_token=continue_token,early_date=date_six_months_ago,later_date=date_six_months_later)
            if "token" in response:
                continue_token = response['token']

            if 'data' not in response:
                # logger.info(f'Fetching {l} data from SemanticScholar.')
                return citation_count, influentionCC
                # raise ConnectionError

            for item in response['data']:
                if int(item['citationCount']) >= 0:
                    citation_count.append(int(item['citationCount']))
                    influentionCC.append(int(item['influentialCitationCount']))
                else:
                    print(item['citationCount'])

        return citation_count, influentionCC

@retry(tries=3)
def get_TNCSI(ref_obj, ref_type='entity', topic_keyword=None,same_year=False,mode=1, show_PDF=False):
    from .semantic_scholar_paper import S2paper
    if ref_type == 'title':
        s2paper = S2paper(ref_obj)
    elif ref_type == 'entity':
        s2paper = ref_obj
    else:
        return None

    if s2paper.citation_count is None:
        rst = {}
        rst['TNCSI'] = -1
        rst['topic'] = 'NONE'
        return rst

    if topic_keyword is None:
        topic = s2paper.gpt_keyword
    else:
        topic = topic_keyword

    if same_year:
        loc, scale = get_PDF_curve_params(topic, topk=1000,  show_img=show_PDF, pub_date=s2paper.publication_date, mode=mode)
    else:
        loc, scale = get_PDF_curve_params(topic, topk=1000, show_img=show_PDF, mode=mode)

    if loc is not None and scale is not None:
        try:
            TNCSI = _get_TNCSI_score(s2paper.citation_count, loc, scale)
        except ZeroDivisionError as e:
            rst = {}
            rst['TNCSI'] = -1
            rst['topic'] = 'NaN'
            return rst
        rst = {}
        rst['TNCSI'] = TNCSI
        rst['topic'] = topic
        rst['loc'] = loc
        rst['scale'] = scale

        return rst
    else:
        rst = {}
        rst['TNCSI'] = -1
        rst['topic'] = topic
        rst['loc'] = loc
        rst['scale'] = scale
        return rst


@retry()
def get_s2citaions_per_month(title, total_num=1000):
    from .semantic_scholar_paper import S2paper
    '''
    :param title: topic keyword
    :param total_num: fetching up to total_num results
    :return: citation count per month
    '''
    s2paper = S2paper(title)

    # Check if the publication date is available for the paper
    if s2paper.publication_date is None:
        print('No publication date recorded')
        return []

    s2id = s2paper.s2id
    citation_count = {}
    missing_count = 0
    OFFSET = 0

    # Construct the URL template for fetching citation data
    url_template = f'https://api.semanticscholar.org/graph/v1/paper/{s2id}/citations?fields=paperId,title,venue,year,referenceCount,citationCount,publicationDate,publicationTypes&offset='


    for i in range(int(total_num / 1000)):
        url = f'{url_template}{OFFSET}&limit=1000'
        OFFSET += 1000

        headers = None
        if s2api is not None:
            headers = {
                'x-api-key': s2api
            }
        r = cached_get(url, headers=headers)

        # Check if the response data is valid
        if 'data' not in r.json() or not r.json()['data']:
            break

        # Process the citation data
        for item in r.json()['data']:
            info = {'paperId': item['citingPaper']['paperId'], 'title': item['citingPaper']['title'],
                    'citationCount': item['citingPaper']['citationCount'],
                    'publicationDate': item['citingPaper']['publicationDate']}

            cite_entry = S2paper(info, ref_type='entity')
            cite_entry.filled_authors = False

            try:
                # Ensure we have valid publication dates before processing
                if s2paper.publication_date <= cite_entry.publication_date and cite_entry.publication_date <= datetime.now():
                    dict_key = f"{cite_entry.publication_date.year}.{cite_entry.publication_date.month}"
                    citation_count[dict_key] = citation_count.get(dict_key, 0) + 1
                else:
                    missing_count += 1
            except Exception as e:
                missing_count += 1
                continue  # Skip this item if there's an issue with publication date

    # Prepare sorted citation data by month
    sorted_data = OrderedDict(
        sorted(citation_count.items(), key=lambda x: datetime.strptime(x[0], "%Y.%m"), reverse=True))

    # Prepare the time range for the result (from earliest publication to the latest month)
    latest_month = datetime.now()
    earliest_month = s2paper.publication_date

    # Generate all months between the earliest and latest months
    all_months = [datetime.strftime(latest_month, "%Y.%#m")]
    while latest_month > earliest_month:
        latest_month = latest_month.replace(day=1)
        latest_month -= timedelta(days=1)
        all_months.append(datetime.strftime(latest_month, "%Y.%#m"))

    # Prepare the result dictionary, filling missing months with 0 citations
    result = {month: sorted_data.get(month, 0) for month in all_months}

    # Sort the result dictionary by month (latest first)
    result = OrderedDict(sorted(result.items(), key=lambda x: datetime.strptime(x[0], "%Y.%m"), reverse=True))

    # Return the final result
    return result


@retry()
def get_IEI(title, show_img=False, save_img_pth=None,exclude_last_n_month=1,normalized=False):
    spms = get_s2citaions_per_month(title, 2000)
    actual_len = 6 if len(spms) >= 6+exclude_last_n_month else len(spms) -exclude_last_n_month
    # print(f'acutal len:{actual_len}')
    if actual_len < 6:
        rst = {}
        rst['L6'] = float('-inf')
        rst['I6'] = float('-inf')
        return rst
    # 
    x = [i for i in range(actual_len)]
    subset = list(spms.items())[exclude_last_n_month:exclude_last_n_month+actual_len][::-1]
    y = [item[1] for item in subset]

    if normalized:
        min_y = min(y)
        max_y = max(y)
        range_y = max_y - min_y
        if range_y == 0:
            y = [0 for _ in y]  # 
        else:
            y = [(y_i - min_y) / range_y for y_i in y]
    # 
    t = np.linspace(0, 1, 100)
    n = len(x) - 1  # 
    curve_x = np.zeros_like(t)
    curve_y = np.zeros_like(t)
    # print(n,y)

    for i in range(n + 1):
        curve_x += comb(n, i) * (1 - t) ** (n - i) * t ** i * x[i]
        curve_y += comb(n, i) * (1 - t) ** (n - i) * t ** i * y[i]
    if show_img or save_img_pth:
        print('IEI绘图执行')
        # 
        plt.clf()
        fig = plt.figure(figsize=(6, 4), dpi=300)  # Increase DPI for high resolution
        plt.style.use('seaborn-v0_8')
        plt.plot(x, y, 'o', color='darkorange', label='Data Point')  # darkorange for contrast
        plt.plot(curve_x, curve_y, color='steelblue', label='Bezier Curve')  # steelblue for the line

        plt.legend()
        plt.xlabel('Month')
        plt.ylabel('Received Citation')
        # plt.title('Quintic Bezier Curve')
        plt.grid(True)
        if save_img_pth:
            plt.savefig(save_img_pth,dpi=300, bbox_inches='tight')
        if show_img:
            plt.show()

    dx_dt = np.zeros_like(t)
    dy_dt = np.zeros_like(t)
    # print(y)
    for i in range(n):
        dx_dt += comb(n - 1, i) * (1 - t) ** (n - i - 1) * t ** i * (x[i + 1] - x[i])
        dy_dt += comb(n - 1, i) * (1 - t) ** (n - i - 1) * t ** i * (y[i + 1] - y[i])

    I6 = dy_dt[-1] / dx_dt[-1]
    # print(len(dy_dt))
    slope_avg = []
    # sum([dy_dt[i-1] / dx_dt[i-1] for i in range(0, 101, 20)])/n+1
    for i in range(0, 100, 20):
        # print((curve_x[i-1],curve_y[i-1]))
        slope_avg.append(dy_dt[i] / dx_dt[i])
    slope_avg.append(I6)
    # print(slope_avg)
    # 
    # 
    rst = {}
    rst['L6'] = sum(slope_avg) / 6 if not math.isnan(sum(slope_avg)) else float('-inf')
    rst['I6'] = I6 if not math.isnan(I6) else float('-inf')
    return rst








def get_pubdate_stat(refs):
    # 
    pub_dates = [i.publication_date for i in refs if i.publication_date and i.publication_date >= datetime(1970, 1, 1)]

    if not pub_dates or len(pub_dates)==0:
        return None

    # 
    timestamps = [d.timestamp() for d in sorted(pub_dates, reverse=True)]

    # 
    median_timestamp = statistics.median(timestamps)
    median_value = datetime.fromtimestamp(median_timestamp)

    return {'med': median_value,'latest':pub_dates[0],"pub_dates":pub_dates}


def get_RQM(ref_obj, ref_type='entity', tncsi_rst=None,beta=20,topic_keyword=None):
    from .semantic_scholar_paper import S2paper

    if ref_type == 'title':
        s2paper = S2paper(ref_obj)
    elif ref_type == 'entity':
        s2paper = ref_obj
    else:
        return None

    if not tncsi_rst:
        tncsi_rst = get_TNCSI(ref_obj, ref_type='entity', topic_keyword=topic_keyword, show_PDF=False)

    loc = tncsi_rst['loc']
    scale = tncsi_rst['scale']


    pub_dates = []
    if len(s2paper.references) == 0 or (loc<0 or scale<0):
        print(f'''Error: Assert len(s2paper.references) == 0 or (loc<0 or scale<0) is Ture.''')
        rst = {}
        rst['RQM'] = None
        rst['ARQ'] = None
        rst['S_mp'] = None
        rst['loc'] = loc
        rst['scale'] = scale
        return rst
    for i in s2paper.references:
        if i.publication_date:
            pub_dates.append(i.publication_date)
    # pub_dates = [i.publication_date for i in s2paper.references]
    sorted_dates = sorted(pub_dates, reverse=True)
    # 
    date_index = len(sorted_dates) // 2

    # 
    index_date = sorted_dates[date_index]

    pub_time = s2paper.publication_date
    months_difference = (pub_time - index_date) // timedelta(days=30)
    S_mp = (months_difference // 6 ) + 1

    N_R = len(s2paper.references)


    score = 0
    for item in s2paper.references:
        try:
            score += _get_TNCSI_score(item.citation_count, loc, scale)  # 1
        except:
            N_R = N_R - 1
            continue
    try:
        ARQ = score / N_R # len(pub_set)
    except ZeroDivisionError:
        ARQ = 0
    rst = {}
    rst['RQM'] = 1 - math.exp(-beta * math.exp(-(1-ARQ) * S_mp))
    rst['ARQ'] = ARQ
    rst['S_mp'] = S_mp
    rst['loc'] = loc
    rst['scale'] = scale
    return rst



def get_RAD(M_pc):
    x = M_pc/12
    coefficients= np.array([-0.0025163,0.00106611, 0.12671325, 0.01288683])

    polynomial_function = np.poly1d(coefficients)
    x_pdf = np.linspace(0, 7.36, 200)  # extended range for PDF
    fitted_y_pdf = polynomial_function(x_pdf)
    pdf_normalized = fitted_y_pdf / np.trapz(fitted_y_pdf, x_pdf)


    # Compute the cumulative probability distribution function (CDF)
    cdf = cumtrapz(pdf_normalized, x_pdf, initial=0)
    cdf = np.where(cdf > 1.0, 1.0, cdf)
    # If x is less than the minimum x value in x_pdf, return 0 (assuming CDF is 0 at negative infinity)
    if x < x_pdf[0]:
        return 0

    # If x is greater than the maximum x value in x_pdf, return 1 (CDF is 1 at positive infinity)
    if x > x_pdf[-1]:
        return 1

    # Find the index in x_pdf that is closest to x
    index = np.searchsorted(x_pdf, x, side="left")

    # Return the corresponding CDF value
    return cdf[index]

def get_RUI(s2paper,p=10,q=10):
    """
    Calculate the integral of a log-normal distribution from 0 to t.

    :param mu: The mean (mu) of the log-normal distribution
    :param sigma: The standard deviation (sigma) of the log-normal distribution
    :param t: The upper limit of the integral
    :return: The integral of the log-normal distribution from 0 to t
    """
    # Calculate the cumulative distribution function (CDF) for log-normal distribution at t



    PC = request_query(s2paper.gpt_keyword,early_date=s2paper.publication_date)

    if s2paper.publication_date:
        ref_median_pubdate = get_pubdate_stat(s2paper.references).get('med')
        if ref_median_pubdate:
            if s2paper.publication_date > ref_median_pubdate:
                t = (datetime.now() - s2paper.publication_date) // timedelta(days=30)
                RAD = get_RAD(t)
                #
                MP = request_query(s2paper.gpt_keyword,early_date=ref_median_pubdate ,later_date=s2paper.publication_date)
                rst = {}
                rst['RAD'] = RAD

                N_pc= PC['total']
                N_mp = MP['total']
                if N_mp == 0:
                    return {'RAD': RAD, 'CDR': float('-inf'), 'RUI': float('-inf')}

                CDR = N_pc / N_mp
                rst['CDR'] = CDR
                rst['RUI'] = p * CDR + q * RAD
                return rst
            else:
                ref_latest_date = get_pubdate_stat(s2paper.references).get('latest')
                print(f'''Caution: Publication date of paper "{s2paper.title}" is no longer accurate. Using the pubdate of the latest reference instead.''')
                t = (datetime.now() - ref_latest_date) // timedelta(days=30)
                RAD = get_RAD(t)

                MP = request_query(s2paper.gpt_keyword, early_date=ref_median_pubdate,later_date=ref_latest_date)
                rst = {}
                rst['RAD'] = RAD

                N_pc = PC['total']
                N_mp = MP['total']
                if N_mp == 0:
                    return {'RAD': RAD, 'CDR': float('-inf'), 'RUI': float('-inf')}

                CDR = N_pc / N_mp
                rst['CDR'] = CDR
                rst['RUI'] = p * CDR + q * RAD
                return rst

    print(f'''Publication date of paper "{s2paper.title}" is not available. This is a remote server issue. ''')
    return {'RAD': None, 'CDR': None, 'RUI': None}




@retry()
def request_query(query,  sort_rule=None,  continue_token=None, early_date: datetime = None, later_date:datetime = None
                  ):# before_pub_date=True
    '''

    :param query:
    :param offset:
    :param limit:
    :param CACHE_FILE:
    :param sort: publicationDate:asc - return oldest papers first.
                citationCount:desc - return most highly-cited papers first.
                paperId - return papers in ID order, low-to-high.
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

    if early_date and later_date is None:
        p_dict['publicationDateOrYear'] = f'{early_date.strftime("%Y-%m-%d")}:'
    elif later_date and early_date is None:
        p_dict['publicationDateOrYear'] = f':{later_date.strftime("%Y-%m-%d")}'
    elif later_date and early_date:
        p_dict['publicationDateOrYear'] = f'{early_date.strftime("%Y-%m-%d")}:{later_date.strftime("%Y-%m-%d")}'
    else:
        pass
    if continue_token:
        p_dict['token'] = continue_token
    if sort_rule:
        p_dict['sort'] = sort_rule
    params = urlencode(p_dict)
    url = (f"{S2_QUERY_URL}?{params}&fields=url,title,abstract,authors,venue,externalIds,referenceCount,"
           f"openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,"
           f"s2FieldsOfStudy,publicationTypes,publicationDate")
    # print(url)
    headers = None
    if s2api is not None:
        headers = {
            'x-api-key': s2api
        }
    reply = cached_get(url,headers=headers)
    response = reply.json()
    if "data" not in response:
        msg = response.get("error") or response.get("message") or "unknown"
        raise Exception(f"error while fetching {reply.url}: {msg}")

    return response


