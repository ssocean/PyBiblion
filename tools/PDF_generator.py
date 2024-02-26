import argparse
import math

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import numpy as np

# 选择热力图颜色映射
cmap = plt.get_cmap('coolwarm')
from furnace.semantic_scholar_paper import S2paper
from tools.gpt_util import get_chatgpt_field
from tools.ref_utils import fit_topic_pdf, _get_TNCSI_score, get_esPubRank

import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics  # 注册字体
from reportlab.pdfbase.ttfonts import TTFont  # 字体类
from reportlab.platypus import Table, SimpleDocTemplate, Paragraph, Image  # 报告内容相关类
from reportlab.lib.pagesizes import letter  # 页面的标志尺寸(8.5*inch, 11*inch)
from reportlab.lib.styles import getSampleStyleSheet  # 文本样式
from reportlab.lib import colors  # 颜色模块
from reportlab.graphics.charts.barcharts import VerticalBarChart  # 图表类
from reportlab.graphics.charts.legends import Legend  # 图例类
from reportlab.graphics.shapes import Drawing  # 绘图工具
from reportlab.lib.units import cm  # 单位：cm
from reportlab.platypus.flowables import Image as RFL
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import os
import openai
openai.api_base = "https://api.chatanywhere.cn"
def make_valid_path(path):
    # 获取当前操作系统的路径分隔符和非法字符列表
    separator = os.path.sep
    invalid_chars = [':', '*', '?', '"', '<', '>', '|']

    # 替换非法字符为路径分隔符
    valid_path = path.replace('/', separator).replace('\\', separator)
    for char in invalid_chars:
        valid_path = valid_path.replace(char, '')

    # 移除开头和结尾的路径分隔符
    valid_path = valid_path.strip(separator)

    # 拼接当前工作目录和合法路径
    valid_path = os.path.join(os.getcwd(), valid_path)
    return valid_path

class ReportLaber:
    # 绘制标题
    @staticmethod
    def draw_title(title: str):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 拿到标题样式
        ct = style['Heading1']
        # 单独设置样式相关属性
        ct.fontSize = 18  # 字体大小
        ct.leading = 50  # 行间距
        ct.textColor = colors.black  # 字体颜色
        ct.alignment = 1  # 居中
        ct.bold = True
        # 创建标题对应的段落，并且返回
        return Paragraph(title, ct)

    # 绘制小标题
    @staticmethod
    def draw_sub_title(title: str):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 拿到标题样式
        ct = style['Heading2']
        # 单独设置样式相关属性
        ct.fontSize = 15  # 字体大小
        ct.leading = 30  # 行间距
        ct.textColor = colors.black  # 字体颜色
        # 创建标题对应的段落，并且返回
        return Paragraph(title, ct)
    @staticmethod
    def draw_min_title(title: str):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 拿到标题样式
        ct = style['Heading3']
        # 单独设置样式相关属性
        ct.fontSize = 15  # 字体大小
        ct.leading = 25  # 行间距
        ct.textColor = colors.black  # 字体颜色
        # 创建标题对应的段落，并且返回
        return Paragraph(title, ct)
    # 绘制普通段落内容
    @staticmethod
    def draw_text(text: str, ct=None):
        # 获取所有样式表
        style = getSampleStyleSheet()
        if ct is None:
            # 获取普通样式
            ct = style['Normal']

            ct.fontSize = 12
            ct.wordWrap = 'CJK'  # 设置自动换行
            ct.alignment = 0  # 左对齐
            ct.firstLineIndent = 32  # 第一行开头空格
            ct.leading = 25
        return Paragraph(text, ct)

    @staticmethod
    def draw_notes(text: str, ct=None):
        # 获取所有样式表
        style = getSampleStyleSheet()
        if ct is None:
            # 获取普通样式
            ct = style['Italic']

            ct.fontSize = 8
            ct.wordWrap = 'CJK'  # 设置自动换行
            ct.alignment = 0  # 左对齐
            ct.firstLineIndent = 32  # 第一行开头空格
            ct.leading = 12
        return Paragraph(text, ct)

    @staticmethod
    def draw_author(sp: S2paper):
        a_names = [str(a.name) for a in sp.authors]
        sub_a_names = [a_names[i:i + 4] for i in range(0, len(a_names), 4)][:]
        author_names = []
        author_bgcs = []
        r = 0
        for i in range(0, len(a_names), 4):

            author_names.append(a_names[i:i + 4])
            for c in range(4):
                try:
                    hindex = sp.authors[r * 4 + c].h_index
                    r_ratio = min(hindex / 100, 1)
                    matlab_color = cmap(r_ratio)

                    author_bgcs.append(('BACKGROUND', (c, r), (c, r),
                                        colors.Color(matlab_color[0], matlab_color[1], matlab_color[2], matlab_color[3])))
                except IndexError:
                    pass
            r += 1

        # 创建表格数据
        table_data = sub_a_names

        # 创建表格对象
        table = Table(table_data, colWidths=100)
        # 表格样式

        table_style = TableStyle([

                                     ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),  # 文本颜色
                                     ('GRID', (0, 0), (-1, -1), 1, colors.black),  # 表格线颜色
                                     ('FONTSIZE', (0, 0), (-1, -1), 10),  # 第一行字体大小
                                 ] + author_bgcs)

        # 创建Table对象，并应用样式

        table.setStyle(table_style)
        # # 设置表格样式
        # table_style = TableStyle([
        #     ('BACKGROUND', (0, 0), (-1, -1), colors.white),  # 设置所有单元格的背景颜色为白色
        # ])
        #
        # for i in range(1, len(table_data)):
        #     bg_color = table_data[i][0]  # 获取每个单元格的背景颜色
        #     table_style.add('BACKGROUND', (0, i), (0, i), bg_color)  # 为每个单元格设置背景颜色
        #
        # table.setStyle(table_style)

        # 将表格添加到文档中
        return table

    # 绘制表格
    @staticmethod
    def draw_table(*args):
        # 列宽度
        col_width = 9.1 * cm
        style = [
            # ('FONTNAME', (0, 0), (-1, -1)),  # 字体
            ('FONTSIZE', (0, 0), (-1, 0), 12),  # 第一行的字体大小
            # ('FONTSIZE', (0, 1), (-1, -1), 10),  # 第二行到最后一行的字体大小
            ('BACKGROUND', (0, 0), (-1, 0), '#d5dae6'),  # 设置第一行背景颜色
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 第一行水平居中
            # ('ALIGN', (0, 1), (-1, -1), 'LEFT'),  # 第二行到最后一行左右左对齐
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.darkslategray),  # 设置表格内文字颜色
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # 设置表格框线为grey色，线宽为0.5
            # ('SPAN', (0, 1), (0, 2)),  # 合并第一列二三行
            # ('SPAN', (0, 3), (0, 4)),  # 合并第一列三四行
            # ('SPAN', (0, 5), (0, 6)),  # 合并第一列五六行
            # ('SPAN', (0, 7), (0, 8)),  # 合并第一列五六行
        ]
        table = Table(args, colWidths=col_width, style=style)
        return table

    # 绘制图片
    @staticmethod
    def draw_img(path):
        img = RFL(path, )  # 读取指定路径下的图片
        img.drawWidth = 9 * cm  # 设置图片的宽度
        img.drawHeight = 6 * cm  # 设置图片的高度
        return img

def plot_ref_aFNCSI(sp,loc,scale):
    importances = []
    aFNCSIs = []
    areas = []
    for i in sp.references:

        ref_time = len(i._entity['contexts'])
        # importance = min(math.log10(ref_time + 1),1)
        importance = math.log(ref_time + 1)
        cite_count = 0 if i.citation_count is None else int(i.citation_count)
        cur_aFNCSI = _get_TNCSI_score(cite_count, loc, scale)
        icite_count = 0 if i.influential_citation_count is None else int(i.citation_count)
        area = math.pi * (5 * math.log10(icite_count + 1)) ** 2

        importances.append(float(importance))
        aFNCSIs.append(cur_aFNCSI)
        areas.append(area)

    x = np.array(importances)
    y = np.array(aFNCSIs)

    colors = np.random.rand(len(sp.references))
    area = np.array(areas)
    plt.clf()

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.xlabel('Influential References Index')

    # 设置 y 轴标签
    plt.ylabel('aFNCSI')
    plt.savefig('plot.png')
def plot_time_vs_aFNCSI(sp:S2paper,loc,scale):
    times = []
    aFNCSIs = []
    areas = []
    sp_pub_date = sp.publication_date
    for i in sp.references:

        ref_time = len(i._entity['contexts'])
        # importance = min(math.log10(ref_time + 1),1)
        if i.publication_date:
            pub_date = i.publication_date
            cite_count = 0 if i.citation_count is None else int(i.citation_count)
            cur_aFNCSI = _get_TNCSI_score(cite_count, loc, scale)
            ref_time = len(i._entity['contexts'])
            # importance = min(math.log10(ref_time + 1),1)
            icite_count = 0 if i.influential_citation_count is None else int(i.citation_count)


            icite_importance = math.log10(icite_count + 1)+1
            ref_importance = math.log(ref_time + 1)+1

            importance = min(icite_importance*ref_importance,50)

            area = math.pi * (importance) ** 2

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
    print(x.shape,y.shape,area.shape)
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.xlabel('Passed Month')

    # 设置 y 轴标签
    plt.ylabel('aFNCSI')
    plt.savefig('plot.png')

def render_pdf(title,field,save_path):
    sp = S2paper(title)
    print(sp.s2id)
    if sp.s2id is None:
        return
    assert sp.s2id is not None, f'Can not fetch anything related to the given title: {title}. This could be caused by the delay or the incorrect title. Please check again or try later.'
    logger.info(f'{sp.title} with s2id {sp.s2id} retrieved successfully')

    if save_path:
        pdf_pth = save_path
    else:
        pdf_pth = title + '_report.pdf'
    pdf_pth = make_valid_path(pdf_pth)
    print(pdf_pth)
    logger.info(f'PDF will be saved to {pdf_pth}')
    doc = SimpleDocTemplate(pdf_pth, pagesize=letter)


    styles = getSampleStyleSheet()

    normal_style = styles["Normal"]
    if field is None:
        logger.info('Research field is not provided, using GPT instead. Make sure that you have set the openai.api_key in util/gpt_util.py')
        topic = get_chatgpt_field(sp.title, sp.abstract)
        topic = topic[0].replace('.','')
    else:
        topic = field
    logger.info(f'Research field is {topic}')
    rl = ReportLaber()

    title = rl.draw_title("Analysis Report")
    subtitle = rl.draw_sub_title(f"Paper Title: {sp.title}")
    if sp.publication_date is not None:
        pub_date = rl.draw_text('<font name="Helvetica-Bold">Date of Publication: </font> ' + str(sp.publication_date.strftime("%Y-%m-%d")), normal_style)
    else:
        pub_date = rl.draw_text(
            '<font name="Helvetica-Bold">Date of Publication is unavailable. </font> ' ,normal_style)
    author = rl.draw_sub_title(f"Authors:")
    author_table = rl.draw_author(sp)
    author_notes = rl.draw_notes("The redder the background colour corresponding to the author's name, the greater the H-Index value for that author\nGrey background indicates that the author's H-Index is temporarily unavailable.")

    if sp.tldr is not None:
        tldr = rl.draw_text('<font name="Helvetica-Bold">TLDR: </font> ' + sp.tldr, normal_style)
    else:
        tldr = rl.draw_text('<font name="Helvetica-Bold">TLDR: </font> TLDR is unavailable.' , normal_style)

    publication_source = rl.draw_text(f"<font name='Helvetica-Bold'>Publication Source:</font>  {sp.publication_source}", normal_style)
    esPubRank = None
    if sp.publication_date:
        esPubRank = get_esPubRank(sp.publication_source)
    rank_notes = None
    if esPubRank:
        rank_notes = ''
        if 'sciif' in esPubRank:
            rank_notes += f"SCI IF: {esPubRank['sciif']}  "
        if 'sci' in esPubRank:
            rank_notes += f"JCR: {esPubRank['sci']}  "
        if 'ccf' in esPubRank:
            rank_notes += f"CCF: {esPubRank['ccf']}  "
        rank_notes = rl.draw_notes(rank_notes)

    field = rl.draw_text(f"<font name='Helvetica-Bold'>Research Field: </font>  {topic}", normal_style)
    citation = rl.draw_text(f"<font name='Helvetica-Bold'>Citation Number: </font>  {sp.citation_count}", normal_style)

    loc, scale= fit_topic_pdf(topic, topk=2000,save_img_pth='pdf.png')

    aFNCSI = _get_TNCSI_score(sp.citation_count, loc, scale)

    aFNCIS_para = rl.draw_text(f"<font name='Helvetica-Bold'>aFNCSI: </font>{aFNCSI}", normal_style)
    aFNCIS_notes = rl.draw_notes('aFNCSI indicates the impact of the paper within its field. The closer the value is to 1, the more influential the paper is.')

    icitation = rl.draw_text(f"<font name='Helvetica-Bold'>Influential Citation Count: </font>{sp.influential_citation_count}", normal_style)

    mini_spacer = Spacer(1, 10)
    spacer = Spacer(1, 20)

    # plot_ref_aFNCSI(sp, loc, scale)
    plot_time_vs_aFNCSI(sp, loc, scale)

    pdf_curve = rl.draw_img('pdf.png')
    FNCSI_scatter = rl.draw_img('plot.png')

    figs = [('Histogram of Citation Frequency Distribution', 'Bubble Chart of References Quality'),
            (pdf_curve, FNCSI_scatter)]

    fig_table = rl.draw_table(*figs)

    fig_notes = rl.draw_notes('<i>*The histogram is based on a maximum of 2,000 statistics. \nLarger bubbles in the bubble chart indicate a higher Influential Citations. The Source data is provided by SemanticScholar.</i>')


    meta_info = [title, subtitle,pub_date, author, author_table, author_notes, mini_spacer, tldr, mini_spacer, field,
                 mini_spacer, publication_source,rank_notes, mini_spacer, citation, icitation, aFNCIS_para,aFNCIS_notes]
    analysis_content = [fig_table,fig_notes]

    content_list = meta_info + [spacer] + analysis_content

    doc.build(content_list)


def main(args):
    sp = S2paper(args.title)
    print(sp.s2id)
    assert sp.s2id is not None, f'Can not fetch anything related to the given title: {args.title}. This could be caused by the delay or the incorrect title. Please check again or try later.'
    logger.info(f'{sp.title} with s2id {sp.s2id} retrieved successfully')

    if args.save_path:
        pdf_pth = args.save_path
    else:
        pdf_pth = args.title + '_report.pdf'
    pdf_pth = make_valid_path(pdf_pth)
    logger.info(f'PDF will be saved to {pdf_pth}')
    doc = SimpleDocTemplate(pdf_pth, pagesize=letter)


    styles = getSampleStyleSheet()

    normal_style = styles["Normal"]
    if args.field is None:
        logger.info('Research field is not provided, using GPT instead. Make sure that you have set the openai.api_key in util/gpt_util.py')
        topic = get_chatgpt_field(sp.title, sp.abstract)
        topic = topic[0].replace('.','')
    else:
        topic = args.field
    logger.info(f'Research field is {topic}')
    rl = ReportLaber()

    title = rl.draw_title("Analysis Report")
    subtitle = rl.draw_sub_title(f"Paper Title: {sp.title}")
    if sp.publication_date is not None:
        pub_date = rl.draw_text('<font name="Helvetica-Bold">Date of Publication: </font> ' + str(sp.publication_date.strftime("%Y-%m-%d")), normal_style)
    else:
        pub_date = rl.draw_text(
            '<font name="Helvetica-Bold">Date of Publication is unavailable. </font> ' ,normal_style)
    author = rl.draw_sub_title(f"Authors:")
    author_table = rl.draw_author(sp)
    author_notes = rl.draw_notes("The redder the background colour corresponding to the author's name, the greater the H-Index value for that author\nGrey background indicates that the author's H-Index is temporarily unavailable.")

    if sp.tldr is not None:
        tldr = rl.draw_text('<font name="Helvetica-Bold">TLDR: </font> ' + sp.tldr, normal_style)
    else:
        tldr = rl.draw_text('<font name="Helvetica-Bold">TLDR: </font> TLDR is unavailable.' , normal_style)

    publication_source = rl.draw_text(f"<font name='Helvetica-Bold'>Publication Source:</font>  {sp.publication_source}", normal_style)


    esPubRank = get_esPubRank(sp.publication_source)
    rank_notes = None
    if esPubRank:
        rank_notes = ''
        if 'sciif' in esPubRank:
            rank_notes += f"SCI IF: {esPubRank['sciif']}  "
        if 'sci' in esPubRank:
            rank_notes += f"JCR: {esPubRank['sci']}  "
        if 'ccf' in esPubRank:
            rank_notes += f"CCF: {esPubRank['ccf']}  "
        rank_notes = rl.draw_notes(rank_notes)

    field = rl.draw_text(f"<font name='Helvetica-Bold'>Research Field: </font>  {topic}", normal_style)
    citation = rl.draw_text(f"<font name='Helvetica-Bold'>Citation Number: </font>  {sp.citation_count}", normal_style)

    loc, scale, image = fit_topic_pdf(topic, topk=2000)
    image.save('pdf.png')
    aFNCSI = _get_TNCSI_score(sp.citation_count, loc, scale)

    aFNCIS_para = rl.draw_text(f"<font name='Helvetica-Bold'>aFNCSI: </font>{aFNCSI}", normal_style)
    aFNCIS_notes = rl.draw_notes('aFNCSI indicates the impact of the paper within its field. The closer the value is to 1, the more influential the paper is.')

    icitation = rl.draw_text(f"<font name='Helvetica-Bold'>Influential Citation Count: </font>{sp.influential_citation_count}", normal_style)

    mini_spacer = Spacer(1, 10)
    spacer = Spacer(1, 20)

    # plot_ref_aFNCSI(sp, loc, scale)
    plot_time_vs_aFNCSI(sp, loc, scale)

    pdf_curve = rl.draw_img('pdf.png')
    FNCSI_scatter = rl.draw_img('plot.png')

    figs = [('Histogram of Citation Frequency Distribution', 'Bubble Chart of References Quality'),
            (pdf_curve, FNCSI_scatter)]

    fig_table = rl.draw_table(*figs)

    fig_notes = rl.draw_notes('<i>*The histogram is based on a maximum of 2,000 statistics. \nLarger bubbles in the bubble chart indicate a higher Influential Citations. The Source data is provided by SemanticScholar.</i>')


    meta_info = [title, subtitle,pub_date, author, author_table, author_notes, mini_spacer, tldr, mini_spacer, field,
                 mini_spacer, publication_source,rank_notes, mini_spacer, citation, icitation, aFNCIS_para,aFNCIS_notes]
    analysis_content = [fig_table,fig_notes]

    content_list = meta_info + [spacer] + analysis_content

    doc.build(content_list)



if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Generate PDF with title, save path, and fields')

    # 添加命令行参数
    parser.add_argument('--title', type=str, help='PDF title', default='New Techniques for Combined FEM-Multibody Anatomical Simulation')
    parser.add_argument('--save_path', type=str, help='PDF save path. If not provided, then the PDF will be saved automatically according to the title', default=None)
    parser.add_argument('--field', type=str, help='Field of research. If not provided, then GPT is used to generate one.',default=None)

    # 解析命令行参数
    args = parser.parse_args()
    main(args)

