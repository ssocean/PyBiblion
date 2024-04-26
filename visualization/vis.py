from nltk import pos_tag
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # 导入PIL模块中的Image对象
import wordcloud  # 导入词云模块
from wordcloud import ImageColorGenerator
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize

nltk.download('averaged_perceptron_tagger')


def read_file(file_name):
    if os.path.exists(file_name):  # 判断文件是否存在
        with open(file_name, "r", encoding='utf-8') as file:  # 读取文件
            content = file.read()
            if content:  # 判断文本内容是否为空
                return content
            else:
                print("文件无内容")
    else:
        print(file_name, "文件不存在")


def generate_word_cloud_illustration(content, figure_name, result_name):
    tokens = word_tokenize(content)
    pos_tags = pos_tag(tokens)
    filtered_pos_tags = [word for word, pos in pos_tags if pos in ['NN', 'NNS']]
    # 构建词频统计
    word_counter = Counter(filtered_pos_tags)
    # not_statis = ['也', '我', '着', '那', '这', '了', '你', ':', '：', '?', '！', '...', '…', '他', '她', '？', '，', '、', '。', '的', '和', '\u3000', '“', '”', ' ', 'ri', '与', '是', '在', '中', '了', '\n']

    # for i in list(word_counter.keys()):
    #     if i in not_statis:
    #         word_counter.pop(i)

    img = np.array(Image.open(figure_name))  # 读取图片
    img_colors = ImageColorGenerator(img)
    wd = wordcloud.WordCloud(mask=img, font_path="simhei.ttf", background_color="white")
    # wd.generate(words)
    print(word_counter)
    wd.generate_from_frequencies(word_counter)
    plt.imshow(wd.recolor(color_func=img_colors), interpolation="bilinear")
    plt.axis("off")
    plt.savefig(result_name)
    plt.show()


# file_name = "story.txt"     # 预读取文本文件名
file_name = 'test.txt'
figure_name = r"visualization\circle.png"  # 词频图形状图片文件名
result_name = 'result.png'  # 保存的词云图文件名
word_statis_name = "word_statistics"  # 词频统计绘制的雷达图文件名
content = read_file(file_name)
generate_word_cloud_illustration(content, figure_name, result_name)
#