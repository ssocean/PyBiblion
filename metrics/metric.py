import logging
import math
import statistics
import time

from config.config import s2api
from furnace.semantic_scholar_paper import S2paper, request_query
from tools.gpt_util import get_chatgpt_field
from tools.ref_utils import _get_TNCSI_score, get_s2citaions_per_month, get_cite_score
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from retry import retry


def fit_topic_pdf(topic, topk=2000, show_img=False, save_img_pth=None):
    import numpy as np
    import matplotlib.pyplot as plt

    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from PIL import Image
    import io




    citation, _ = _plot_s2citaions(topic, total_num=1000)
    citation = np.array(citation)
    # 绘制直方图
    # plt.hist(citation, bins=1000, density=False, alpha=0.7, color='skyblue')
    # # 添加标题和标签
    # plt.title('Distribution of Data')
    # plt.xlabel('Value')
    # plt.ylabel('Density')
    # 显示图形
    # plt.show()

    # 拟合数据到指数分布
    try:
        params = stats.expon.fit(citation)
    except:
        # print(citation)
        return None,None
    loc, scale = params

    # 生成拟合的指数分布曲线
    x = np.linspace(np.min(citation), np.max(citation), 100)
    pdf = stats.expon.pdf(x, loc, scale)

    # 绘制原始数据和拟合的指数分布曲线
    if show_img or save_img_pth:
        plt.clf()
        plt.figure(figsize=(6, 4))
        plt.hist(citation, bins=1000, density=True, alpha=0.5)
        plt.plot(x, pdf, 'r', label='Fitted Exponential Distribution')
        plt.xlabel('Number of Citations')
        # 设置 y 轴标签
        plt.ylabel('Frequency')
        plt.legend()
        if save_img_pth:
            print('saving success')
            plt.savefig(save_img_pth)
        plt.show()

        # # 将图形渲染到画布
        # canvas = FigureCanvas(plt.gcf())
        # buffer = io.BytesIO()
        # canvas.print_png(buffer)
        #
        # # 将画布转换为PIL图像对象
        # buffer.seek(0)
        # image = Image.open(buffer)

    # # 利用拟合的参数重新绘制概率密度函数图形
    # plt.plot(x, stats.expon.pdf(x, loc, scale), 'b', label='Fitted PDF')
    # plt.legend()
    # plt.show()
    # from PIL import Image
    #
    # # 假设您已经有了图像数据 image_data
    #
    # # 创建 PIL 图像对象
    # image = Image.fromarray(image)

    # 保存为 PNG 文件
    # image.save("output.png")
    return loc, scale  # , image
# @retry(tries=3)
def _plot_s2citaions(keyword: str, year: str = None, total_num=2000, CACHE_FILE='.ppicache'):
    '''

    :param keyword: topic keyword
    :param year: like 2018-2023 || 2018
    :param total_num:  fetching up to total_num results
    :param CACHE_FILE:
    :return:
    '''
    l = 0
    citation_count = []
    influentionCC = []
    with shelve.open(CACHE_FILE) as cache:
        for i in range(int(total_num / 100)):
            if year:
                url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&fieldsOfStudy=Computer Science&year={year}&fields=title,year,citationCount,influentialCitationCount&offset={100 * i}&limit=100'
            else:
                url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&fieldsOfStudy=Computer Science&fields=title,year,citationCount,influentialCitationCount&offset={100 * i}&limit=100'

            if url in cache:
                r = cache[url]
            else:

                if s2api is not None:
                    headers = {
                        'x-api-key': s2api
                    }
                else:
                    headers = None
                r = requests.get(url, headers=headers,verify=False)
                time.sleep(0.5)
                cache[url] = r


            # print(r.json())

            try:
                if 'data' not in r.json():
                    # logger.info(f'Fetching {l} data from SemanticScholar.')
                    return citation_count, influentionCC
                    # raise ConnectionError

                for item in r.json()['data']:
                    if int(item['citationCount']) >= 0:
                        citation_count.append(int(item['citationCount']))
                        influentionCC.append(int(item['influentialCitationCount']))
                        l += 1
                    else:
                        print(item['citationCount'])
            except:
                continue

        # logger.info(f'Fetch {l} data from SemanticScholar.')
    return citation_count, influentionCC
@retry(tries=3)
def get_TNCSI(ref_obj, ref_type='entity', topic_keyword=None, save_img_pth=None,show_PDF=False):
    if ref_type == 'title':
        s2paper = S2paper(ref_obj)
    elif ref_type == 'entity':
        s2paper = ref_obj
    else:
        return None

    if topic_keyword is None:

        topic = get_chatgpt_field(ref_obj, s2paper.abstract)
        topic = topic[0].replace('.', '')
        logger.info(
            f'Generated research field is {topic}.')
    else:
        topic = topic_keyword
        # logger.info(f'Pre-defined research field is {topic}')

    loc, scale = fit_topic_pdf(topic, topk=2000,save_img_pth=save_img_pth,show_img=show_PDF)
    if loc is not None and scale is not None:
        TNCSI = _get_TNCSI_score(s2paper.citation_count, loc, scale)
        rst = {}
        rst['TNCSI'] = TNCSI
        rst['topic'] = topic
        rst['loc'] = loc
        rst['scale'] = scale

        return rst
    else:
        raise FileNotFoundError


from retry import retry
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

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
    # 六个坐标点
    x = [i for i in range(actual_len)]
    subset = list(spms.items())[exclude_last_n_month:exclude_last_n_month+actual_len][::-1]
    y = [item[1] for item in subset]
    if normalized:
        y = [(y_i - min(y)) / (max(y) - min(y)) for y_i in y]
    # 拟合五次贝塞尔曲线
    t = np.linspace(0, 1, 100)
    n = len(x) - 1  # 控制点的数量
    curve_x = np.zeros_like(t)
    curve_y = np.zeros_like(t)
    # print(n,y)

    for i in range(n + 1):
        curve_x += comb(n, i) * (1 - t) ** (n - i) * t ** i * x[i]
        curve_y += comb(n, i) * (1 - t) ** (n - i) * t ** i * y[i]
    if show_img or save_img_pth:
        # 绘制曲线
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
    print(slope_avg)
    print('平均', sum(slope_avg) / 6)
    print("瞬时斜率:", I6)
    rst = {}
    rst['L6'] = sum(slope_avg) / 6 if not math.isnan(sum(slope_avg)) else float('-inf')
    rst['I6'] = I6 if not math.isnan(I6) else float('-inf')
    return rst



S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
CACHE_FILE = r"C:\Users\Ocean\Documents\GitHub\Dynamic_Literature_Review\CACHE\.queryCache"
#
#
# def request_query(query, CACHE_FILE=CACHE_FILE, sort_rule=None, pub_date: datetime = None, continue_token=None):
#     '''
#
#     :param query:
#     :param offset:
#     :param limit:
#     :param CACHE_FILE:
#     :param sort: publicationDate:asc - return oldest papers first.
#                 citationCount:desc - return most highly-cited papers first.
#                 paperId - return papers in ID order, low-to-high.
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
#     if pub_date:
#         p_dict['publicationDateOrYear'] = f':{pub_date.year}-{pub_date.month}-{pub_date.day}'
#     if continue_token:
#         p_dict['token'] = continue_token
#     if sort_rule:
#         p_dict['sort'] = sort_rule
#     params = urlencode(p_dict)
#     url = (f"{S2_QUERY_URL}?{params}&fields=url,title,abstract,authors,venue,externalIds,referenceCount,"
#            f"openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,"
#            f"s2FieldsOfStudy,publicationTypes,publicationDate")
#     with shelve.open(generate_cache_file_name(url)) as cache:
#
#         # if pub_date:
#         #     url = url+f'$publicationDateOrYear=:{pub_date.year}-{pub_date.month}-{pub_date.day}'
#         # if continue_token:
#         #     url = url+f'$token={continue_token}'
#         # print(url)
#         if url in cache:
#             reply = cache[url]
#         else:
#             session = requests.Session()
#             if s2api is not None:
#                 headers = {
#                     'x-api-key': s2api
#                 }
#             else:
#                 headers = None
#             reply = session.get(url, headers=headers)
#             cache[url] = reply
#
#             reply = session.get(url)
#         response = reply.json()
#
#         if "data" not in response:
#             msg = response.get("error") or response.get("message") or "unknown"
#             raise Exception(f"error while fetching {reply.url}: {msg}")
#
#         return response


from CACHE.CACHE_Config import generate_cache_file_name
from tools.gpt_util import get_chatgpt_fields

import requests
from urllib.parse import urlencode
import shelve

S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
CACHE_FILE = r"C:\Users\Ocean\Documents\GitHub\Dynamic_Literature_Review\CACHE\.queryCache"
@retry()
def request_query(query,  sort_rule=None,  continue_token=None, early_date: datetime.datetime = None, later_date:datetime.datetime = None
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
        # if before_pub_date:
        #
        # else:
        #
    if continue_token:
        p_dict['token'] = continue_token
    if sort_rule:
        p_dict['sort'] = sort_rule
    params = urlencode(p_dict)
    url = (f"{S2_QUERY_URL}?{params}&fields=url,title,abstract,authors,venue,externalIds,referenceCount,"
           f"openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,"
           f"s2FieldsOfStudy,publicationTypes,publicationDate")
    # print(url)
    with shelve.open(generate_cache_file_name(url)) as cache:

        # if pub_date:
        #     url = url+f'$publicationDateOrYear=:{pub_date.year}-{pub_date.month}-{pub_date.day}'
        # if continue_token:
        #     url = url+f'$token={continue_token}'
        # print(url)
        if url in cache:
            reply = cache[url]
            if reply.status_code == 504:
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

def get_Coverage(ref_obj, ref_type='entity', tncsi_rst=None, multiple_keywords=False):
    def cal_weighted_cocite_rate(ref_relevant_lst, pub_relevant_lst,tncsi_rst):
        '''

        :param ref_list_Anchor:
        :param ref_list_eval:
        :return:
        '''
        loc = tncsi_rst['loc']
        scale = tncsi_rst['scale']
        ref_list = [(i.title,i.citation_count) for i in ref_relevant_lst]
        pub_list = [(i.title,i.citation_count) for i in pub_relevant_lst]

        # print(len(ref_list))
        # print(len(pub_list))
        ref_set = set(ref_list)
        pub_set = set(pub_list)
        # print()
        # print(len(ref_list))
        # print(len(pub_set))

        intersection = ref_set & pub_set
        intersection = list(intersection)
        # print(anchor_set)
        #
        #
        # print(eval_set)

        # print(len(intersection))

        # exclude = ref_set - set(intersection)
        # print(exclude)
        # print(len(exclude))
        # print(intersection)
        score = 0
        print(f'raw coverage:{len(intersection)/len(pub_set)}')
        for item in intersection:
            score += _get_TNCSI_score(item[-1],loc,scale)#1
        try:
            score_of_relevant = sum([_get_TNCSI_score(i[-1],loc,scale) for i in list(pub_set)])
            overlap_ratio = score / score_of_relevant#len(pub_set)
        except ZeroDivisionError:
            return 0
        return overlap_ratio

    if ref_type == 'title':
        s2paper = S2paper(ref_obj)
    elif ref_type == 'entity':
        s2paper = ref_obj
    else:
        return None

    if tncsi_rst:
        loc = tncsi_rst['loc']
        scale = tncsi_rst['scale']
    else:
        print('TODO')
        return

    C = int(2 * (math.log(0.01, math.e) * (-1 * scale) + loc))

    # get Reference_relevant
    ref_r = s2paper.references
    print(f'search paper title:{s2paper.title}, which has {len(ref_r)} refs')
    sort_rule = 'citationCount:desc'  # 'citationCount:desc' 'publicationDate:desc'

    # get Publications_relevant
    pub_r = []
    continue_token = None
    if multiple_keywords:
        keywords = s2paper.gpt_keywords
        # keywords = ['Instance segmentation','panoptic segmentation','semantic segmentation','weakly supervised','unsupervised','domain adaptation']
        # keywords = ['curriculum learning', 'self-paced learning', 'training strategy']
        keywords = [f'"{i}"~25' for i in keywords]
        print(keywords)
        topic = ' | '.join(keywords)
        for i in range(0, C, 1000):
            if continue_token is None:
                response = request_query(topic, sort_rule=sort_rule, pub_date=s2paper.publication_date)
            else:
                response = request_query(topic, sort_rule=sort_rule, pub_date=s2paper.publication_date,
                                         continue_token=continue_token)
            # print(response.keys())

            if "token" in response:
                continue_token = response['token']

            if "data" not in response:
                msg = response.get("error") or response.get("message") or "unknown"
                logger.warning('No matched paper!')
                # raise Exception(f"error while fetching {reply.url}: {msg}")
            else:
                for entity in response['data']:
                    temp_ref = S2paper(entity, ref_type='entity', force_return=False, filled_authors=False)
                    pub_r.append(temp_ref)

    else:
        for i in range(0, C, 1000):
            if continue_token is None:
                response = request_query(tncsi_rst['topic'], sort_rule=sort_rule, pub_date=s2paper.publication_date)
            else:
                response = request_query(tncsi_rst['topic'], sort_rule=sort_rule, pub_date=s2paper.publication_date,
                                         continue_token=continue_token)
            # print(response.keys())
            if "token" in response:
                continue_token = response['token']
            if "data" not in response:
                msg = response.get("error") or response.get("message") or "unknown"
                logger.warning('No matched paper!')
                # raise Exception(f"error while fetching {reply.url}: {msg}")
            else:
                for entity in response['data']:
                    temp_ref = S2paper(entity, ref_type='entity', force_return=False, filled_authors=False)
                    pub_r.append(temp_ref)

    print(f'Success retrieving {len(pub_r)}/{C} related work, ')

    # print(len(pub_r))

    coverage = cal_weighted_cocite_rate(ref_r, pub_r[:C],tncsi_rst)
    return coverage

# def get_Timeliness(s2paper):
#     import statistics
#     import datetime,math
#
#     pub_dates = []
#     for i in s2paper.references:
#         if i.publication_date:
#             pub_dates.append(i.publication_date)
#     # pub_dates = [i.publication_date for i in s2paper.references]
#     sorted_list = sorted(pub_dates, reverse=True)
#     index = 0
#     while index < len(sorted_list):
#         try:
#             sorted_list[index].timestamp()
#             index += 1
#         except Exception as e:
#             print(f"Error: {e}")
#             del sorted_list[index]
#
#     timestamps = [d.timestamp() for d in sorted_list]
#     if len(timestamps) == 0:
#         return float('-inf')
#     median_timestamp = statistics.median(timestamps)
#     median_value = datetime.datetime.fromtimestamp(median_timestamp)
#
#     # median_value = statistics.median(sorted_list)
#     pub_time = s2paper.publication_date
#     months_difference = (pub_time - median_value) // datetime.timedelta(days=30)
#     print(months_difference)
#     alpha = 0.5
#     timeliness = 1 / (1 + alpha*math.log(1+months_difference,math.e))
#     timeliness = 1 / (1 + alpha*math.sqrt(1+months_difference**2))
#     # print(months_difference, timeliness)
#     return timeliness


from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def get_median_pubdate(pub_time,refs):
    pub_dates = []
    for i in refs:
        if i.publication_date:
            pub_dates.append(i.publication_date)
    # pub_dates = [i.publication_date for i in s2paper.references]
    sorted_list = sorted(pub_dates, reverse=True)
    index = 0
    while index < len(sorted_list):
        try:
            sorted_list[index].timestamp()
            index += 1
        except Exception as e:
            print(f"Error: {e}")
            del sorted_list[index]

    timestamps = [d.timestamp() for d in sorted_list]
    if len(timestamps) == 0:
        return float('-inf')
    median_timestamp = statistics.median(timestamps)
    median_value = datetime.datetime.fromtimestamp(median_timestamp)

    # median_value = statistics.median(sorted_list)
    # months_difference = (pub_time - median_value) // datetime.timedelta(days=30)
    return median_value
# 根据面积值选择冷色系或暖色系

def plot_time_vs_aFNCSI(sp: S2paper, loc, scale):
    def create_cool_colors():
        colors = ["#D6EDFF", "#B1DFFF", "#8CCBFF", "#66B8FF", "#FF69B4",
                  "#1D8BFF", "#0077E6", "#005CBF", "#00408C", "#002659"]

        return random.choice(colors)

    def create_warm_colors():
        colors = ["#FFD6D6", "#FFB1B1", "#FF8C8C", "#FF6666", "#FF4040",
                  "#FF1D1D", "#E60000", "#BF0000", "#8C0000", "#590000"]

        return random.choice(colors)

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # 创建冷色系
    cool_cmap = create_cool_colors()

    # 创建暖色系
    warm_cmap = create_warm_colors()
    times = []
    aFNCSIs = []
    areas = []
    sp_pub_date = sp.publication_date
    for i in tqdm(sp.references):

        ref_time = len(i._entity['contexts'])
        # importance = min(math.log10(ref_time + 1),1)
        if i.publication_date:
            pub_date = i.publication_date
            cite_count = 0 if i.citation_count is None else int(i.citation_count)
            cur_aFNCSI = _get_TNCSI_score(cite_count, loc, scale)
            # ref_time = len(i._entity['contexts'])
            # importance = min(math.log10(ref_time + 1),1)
            # icite_count = 0 if i.influential_citation_count is None else int(i.citation_count)

            # icite_importance = math.log10(icite_count + 1)+1
            # ref_importance = math.log(ref_time + 1)+1

            # importance = min(icite_importance*ref_importance,50)
            temp_IEI = get_IEI(i.title)['L6']
            temp_IEI = sigmoid(temp_IEI)

            temp_r = ((temp_IEI * 2) + 1) ** 2

            if temp_IEI < 0.5:
                print(get_IEI(i.title)['L6'], math.pi * (temp_r) ** 2)
            area = math.pi * ((temp_r) ** 2)

            start_year = pub_date.year
            start_month = pub_date.month
            end_year = sp_pub_date.year
            end_month = sp_pub_date.month

            diff_month = (end_year - start_year) * 12 + (end_month - start_month)

            times.append(diff_month)
            aFNCSIs.append(cur_aFNCSI)
            areas.append(area)

    x = np.array(times)
    y = np.array(aFNCSIs)
    colors = np.random.rand(x.shape[0])
    area = np.array(areas)

    plt.clf()

    plt.figure(figsize=(6, 4))
    # print(x.shape,y.shape,area.shape)
    colors = []
    for a in area:
        if a >= math.pi * (4) ** 2:
            colors.append(create_warm_colors())
        else:
            colors.append(create_cool_colors())

    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.xlabel('Month Before Publication')

    # 设置 y 轴标签
    plt.ylabel('aTNCSI')
    plt.savefig(f'{sp.title}.svg')
# import datetime

# @retry()
# def get_RAI(s2paper):
#     response = request_query(s2paper.gpt_keyword, CACHE_FILE=CACHE_FILE,sort_rule='citationCount:desc',pub_date=s2paper.publication_date,before_pub_date=False)
#     N_p = response['total']
#     N_r = s2paper.reference_count
#     if N_r == 0:
#         return float('-inf')
#     pub_time = s2paper.publication_date
#     current_time = datetime.datetime.now()
#     M_mc = (current_time - pub_time) // datetime.timedelta(days=30)
#
#
#     RAI = math.log(1+M_mc,math.e)*(N_p/N_r)
#     return RAI

# get_Coverage(ref_obj, ref_type='entity', tncsi_rst=None)
# get_IEI('Self-regulating Prompts: Foundational Model Adaptation without Forgetting',True)

def _get_RQM(ref_obj, ref_type='entity', tncsi_rst=None):


    if ref_type == 'title':
        s2paper = S2paper(ref_obj)
    elif ref_type == 'entity':
        s2paper = ref_obj
    else:
        return None

    if not tncsi_rst:
        tncsi_rst = get_TNCSI(ref_obj, ref_type='entity', topic_keyword=None, save_img_pth=None,show_PDF=False)

    loc = tncsi_rst['loc']
    scale = tncsi_rst['scale']




    # get Reference_relevant
    ref_r = s2paper.references
    N_R = len(ref_r)


    score = 0
    for item in ref_r:
        try:
            score += _get_TNCSI_score(item.citation_count, loc, scale)  # 1
        except:
            N_R = N_R - 1
            continue
    try:

        overlap_ratio = score / N_R # len(pub_set)
    except ZeroDivisionError:
        return 0

    # print(f'search paper title:{s2paper.title}, which has {len(ref_r)} refs. Due to errors, only count {N_R} papers.')
    return overlap_ratio

def get_RQM(ref_obj, ref_type='entity', tncsi_rst=None,beta=20):


    if ref_type == 'title':
        s2paper = S2paper(ref_obj)
    elif ref_type == 'entity':
        s2paper = ref_obj
    else:
        return None

    if not tncsi_rst:
        tncsi_rst = get_TNCSI(ref_obj, ref_type='entity', topic_keyword=None, save_img_pth=None,show_PDF=False)

    loc = tncsi_rst['loc']
    scale = tncsi_rst['scale']


    pub_dates = []
    for i in s2paper.references:
        if i.publication_date:
            pub_dates.append(i.publication_date)
    # pub_dates = [i.publication_date for i in s2paper.references]
    sorted_dates = sorted(pub_dates, reverse=True)
    # index = 0
    # while index < len(sorted_list):
    #     try:
    #         sorted_list[index].timestamp()
    #         index += 1
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         del sorted_list[index]
    # 计算前1/3处的索引位置
    date_index = len(sorted_dates) // 2

    # 取前1/3处的日期
    index_date = sorted_dates[date_index]

    # timestamps = [d.timestamp() for d in sorted_list]
    # if len(timestamps) == 0:
    #     return float('-inf')
    # median_timestamp = statistics.median(timestamps)
    # median_value = datetime.datetime.fromtimestamp(median_timestamp)
    # median_value = statistics.median(sorted_list)
    pub_time = s2paper.publication_date
    months_difference = (pub_time - index_date) // datetime.timedelta(days=30)
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
    return rst

import numpy as np
import scipy.stats as stats
from scipy.integrate import cumtrapz
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

def get_RUI(s2paper,p=10,q=10, M=None):
    """
    Calculate the integral of a log-normal distribution from 0 to t.

    :param mu: The mean (mu) of the log-normal distribution
    :param sigma: The standard deviation (sigma) of the log-normal distribution
    :param t: The upper limit of the integral
    :return: The integral of the log-normal distribution from 0 to t
    """
    # Calculate the cumulative distribution function (CDF) for log-normal distribution at t
    t = (datetime.datetime.now() - s2paper.publication_date) // datetime.timedelta(days=30)
    # RAD = stats.lognorm.cdf(t, s=sigma, scale=np.exp(mu))
    RAD = get_RAD(t)
    PC = request_query(s2paper.gpt_keyword,early_date=s2paper.publication_date)
    M = datetime.datetime.now() if not M else M
    MP = request_query(s2paper.gpt_keyword,early_date=get_median_pubdate(M,s2paper.references),later_date=s2paper.publication_date)
    rst = {}
    rst['RAD'] = RAD

    N_pc= PC['total']
    N_mp = MP['total']
    if N_mp == 0:
        return {'RAD':RAD, 'CDR':float('-inf'),'RUI':float('-inf')}

    CDR = N_pc/N_mp
    rst['CDR'] = CDR
    rst['RUI'] = p*CDR + q*RAD
    return rst
# s2paper = S2paper('Image segmentation using deep learning: A survey')
# rqm = get_RQM(s2paper, ref_type='entity')
# print(rqm)