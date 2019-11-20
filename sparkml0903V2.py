# -*- coding: UTF-8 -*-
import sys,time
import math
from operator import add
from pyspark.sql import SparkSession,SQLContext,Row
#from pytoolkit import TDWSQLProvider
#from pytoolkit import TableDesc, TDWUtil
#from pyspark.ml.feature import Word2Vec
#from word2vec import Word2Vec
from multiprocessing.dummy import Queue
from math import log as math_log
#import re
import logging
from itertools import chain
from collections import defaultdict, Iterable
import os
import jieba
print('Import Jieba')

    
    
from nlp_zero import *
print('Import NLP_zero')    
import AhoCorasickTree as ACT
print('Import ACT')


from pyspark import SparkContext

from pytoolkit import TDWSQLProvider
from pytoolkit import TableDesc, TDWUtil

from pyspark.sql.types import StructField,StringType,ArrayType,StructType
from pyspark.sql.functions import udf


print('Import Word2vec')




import re
print('Import re')
import pandas as pd
print('Import pd')
import numpy as np
print('Import np')
#import pymongo
#import logging

import codecs
print('Import codecs')
from multiprocessing.dummy import Queue
print('Import multiprocessing')

#########################################################################################################################
import networkx as nx
import os

try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass
    
sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']

PY2 = sys.version_info[0] == 2
if not PY2:
    # Python 3.x and up
    text_type    = str
    string_types = (str,)
    xrange       = range

    def as_text(v):  ## 生成unicode字符串
        if v is None:
            return None
        elif isinstance(v, bytes):
            return v.decode('utf-8', errors='ignore')
        elif isinstance(v, str):
            return v
        else:
            raise ValueError('Unknown type %r' % type(v))

    def is_text(v):
        return isinstance(v, text_type)

else:
    # Python 2.x
    text_type    = unicode
    string_types = (str, unicode)
    xrange       = xrange

    def as_text(v):
        if v is None:
            return None
        elif isinstance(v, unicode):
            return v
        elif isinstance(v, str):
            return v.decode('utf-8', errors='ignore')
        else:
            raise ValueError('Invalid type %r' % type(v))

    def is_text(v):
        return isinstance(v, text_type)

__DEBUG = None

def debug(*args):
    global __DEBUG
    if __DEBUG is None:
        try:
            if os.environ['DEBUG'] == '1':
                __DEBUG = True
            else:
                __DEBUG = False
        except:
            __DEBUG = False
    if __DEBUG:
        print( ' '.join([str(arg) for arg in args]) )

class AttrDict(dict):
    """Dict that can get attribute by dot"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def combine(word_list, window = 2):
    """构造在window下的单词组合，用来构造单词之间的边。
    
    Keyword arguments:
    word_list  --  list of str, 由单词组成的列表。
    windows    --  int, 窗口大小。
    """
    if window < 2: window = 2
    for x in xrange(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2)
        for r in res:
            yield r

def get_similarity(word_list1, word_list2):
    """默认的用于计算两个句子相似度的函数。

    Keyword arguments:
    word_list1, word_list2  --  分别代表两个句子，都是由单词组成的列表
    """
    words   = list(set(word_list1 + word_list2))        
    vector1 = [float(word_list1.count(word)) for word in words]
    vector2 = [float(word_list2.count(word)) for word in words]
    
    vector3 = [vector1[x]*vector2[x]  for x in xrange(len(vector1))]
    vector4 = [1 for num in vector3 if num > 0.]
    co_occur_num = sum(vector4)

    if abs(co_occur_num) <= 1e-12:
        return 0.
    
    denominator = math.log(float(len(word_list1))) + math.log(float(len(word_list2))) # 分母
    
    if abs(denominator) < 1e-12:
        return 0.
    
    return co_occur_num / denominator

def sort_words(vertex_source, edge_source, window = 2, pagerank_config = {'alpha': 0.85,}):
    """将单词按关键程度从大到小排序

    Keyword arguments:
    vertex_source   --  二维列表，子列表代表句子，子列表的元素是单词，这些单词用来构造pagerank中的节点
    edge_source     --  二维列表，子列表代表句子，子列表的元素是单词，根据单词位置关系构造pagerank中的边
    window          --  一个句子中相邻的window个单词，两两之间认为有边
    pagerank_config --  pagerank的设置
    """
    sorted_words   = []
    word_index     = {}
    index_word     = {}
    _vertex_source = vertex_source
    _edge_source   = edge_source
    words_number   = 0
    for word_list in _vertex_source:
        for word in word_list:
            if not word in word_index:
                word_index[word] = words_number
                index_word[words_number] = word
                words_number += 1

    nx_graph=nx.Graph()
    for word_list in _edge_source:
        for w1, w2 in combine(word_list, window):
            if w1 in word_index and w2 in word_index:
                index1 = word_index[w1]
                index2 = word_index[w2]
                nx_graph.add_edge(index1,index2)

    #print('Graph Initialization Finished!')


    scores = nx.pagerank(nx_graph, **pagerank_config)          # this is a dict
    sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)
    for index, score in sorted_scores:
        item = AttrDict(word=index_word[index], weight=score)
        sorted_words.append(item)

    return sorted_words

def sort_sentences(sentences, words, sim_func = get_similarity, pagerank_config = {'alpha': 0.85,}):
    """将句子按照关键程度从大到小排序

    Keyword arguments:
    sentences         --  列表，元素是句子
    words             --  二维列表，子列表和sentences中的句子对应，子列表由单词组成
    sim_func          --  计算两个句子的相似性，参数是两个由单词组成的列表
    pagerank_config   --  pagerank的设置
    """
    sorted_sentences = []
    _source = words
    sentences_num = len(_source)        
    graph = np.zeros((sentences_num, sentences_num))
    
    for x in xrange(sentences_num):
        for y in xrange(x, sentences_num):
            similarity = sim_func( _source[x], _source[y] )
            graph[x, y] = similarity
            graph[y, x] = similarity
            
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph, **pagerank_config)              # this is a dict
    sorted_scores = sorted(scores.items(), key = lambda item: item[1], reverse=True)

    for index, score in sorted_scores:
        item = AttrDict(index=index, sentence=sentences[index], weight=score)
        sorted_sentences.append(item)

    return sorted_sentences





######################################################################################
#import jieba.posseg as pseg
#from jieba import posseg as pesg
#import codecs



def get_default_stop_words_file():
    tdw2 = TDWSQLProvider(session, user="tdw_yorksywang", passwd="bq0602BQB", db="fkana_db", group = 'cft')
    df = tdw2.table(tblName = 'york_stopwords_textrank')
    
    #rdd = df..rdd.map(lambda row : row[0])



    return df.select('txt').rdd.flatMap(lambda x: x).collect()

class WordSegmentation(object):
    """ 分词 """
    
    def __init__(self, stop_words_file = None, allow_speech_tags = allow_speech_tags):
        """
        Keyword arguments:
        stop_words_file    -- 保存停止词的文件路径，utf8编码，每行一个停止词。若不是str类型，则使用默认的停止词
        allow_speech_tags  -- 词性列表，用于过滤
        """     
        
        allow_speech_tags = [as_text(item) for item in allow_speech_tags]

        self.default_speech_tag_filter = allow_speech_tags
        self.stop_words = set()
        #self.stop_words_file = get_default_stop_words_file()
        #print(self.stop_words_file)
        #if type(stop_words_file) is str:
        #    self.stop_words_file = stop_words_file
        for word in get_default_stop_words_file():
            self.stop_words.add(word.strip())
    
    def segment(self, text, lower = True, use_stop_words = True, use_speech_tags_filter = False):
        """对一段文本进行分词，返回list类型的分词结果

        Keyword arguments:
        lower                  -- 是否将单词小写（针对英文）
        use_stop_words         -- 若为True，则利用停止词集合来过滤（去掉停止词）
        use_speech_tags_filter -- 是否基于词性进行过滤。若为True，则使用self.default_speech_tag_filter过滤。否则，不过滤。    
        """
        text = as_text(text)
        jieba_result = jieba.lcut(text)
                        

        if use_speech_tags_filter == True:
            jieba_result = [w for w in jieba_result]
        else:
            jieba_result = [w for w in jieba_result]

        # 去除特殊符号
        word_list = [w.strip() for w in jieba_result]
        word_list = [word for word in word_list if len(word)>0]
        
        if lower:
            word_list = [word.lower() for word in word_list]

        if use_stop_words:
            word_list = [word.strip() for word in word_list if word.strip() not in self.stop_words]

        return word_list
        
    def segment_sentences(self, sentences, lower=True, use_stop_words=True, use_speech_tags_filter=False):
        """将列表sequences中的每个元素/句子转换为由单词构成的列表。
        
        sequences -- 列表，每个元素是一个句子（字符串类型）
        """
        
        res = []
        for sentence in sentences:
            res.append(self.segment(text=sentence, 
                                    lower=lower, 
                                    use_stop_words=use_stop_words, 
                                    use_speech_tags_filter=use_speech_tags_filter))
        return res
        
class SentenceSegmentation(object):
    """ 分句 """
    
    def __init__(self, delimiters=sentence_delimiters):
        """
        Keyword arguments:
        delimiters -- 可迭代对象，用来拆分句子
        """
        self.delimiters = set([as_text(item) for item in delimiters])
    
    def segment(self, text):
        res = [as_text(text)]
        
        debug(res)
        debug(self.delimiters)

        for sep in self.delimiters:
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if len(s.strip()) > 0]
        return res 
        
class Segmentation(object):
    
    def __init__(self, stop_words_file = None, 
                    allow_speech_tags = allow_speech_tags,
                    delimiters = sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file -- 停止词文件
        delimiters      -- 用来拆分句子的符号集合
        """
        self.ws = WordSegmentation(stop_words_file=stop_words_file, allow_speech_tags=allow_speech_tags)
        self.ss = SentenceSegmentation(delimiters=delimiters)
        
    def segment(self, text, lower = False):
        text = as_text(text)
        sentences = self.ss.segment(text)
        words_no_filter = self.ws.segment_sentences(sentences=sentences, 
                                                    lower = lower, 
                                                    use_stop_words = False,
                                                    use_speech_tags_filter = False)
        words_no_stop_words = self.ws.segment_sentences(sentences=sentences, 
                                                    lower = lower, 
                                                    use_stop_words = True,
                                                    use_speech_tags_filter = False)

        words_all_filters = self.ws.segment_sentences(sentences=sentences, 
                                                    lower = lower, 
                                                    use_stop_words = True,
                                                    use_speech_tags_filter = True)

        return AttrDict(
                    sentences           = sentences, 
                    words_no_filter     = words_no_filter, 
                    words_no_stop_words = words_no_stop_words, 
                    words_all_filters   = words_all_filters
                )




######################################################################################

class TextRank4Keyword(object):
    
    def __init__(self, stop_words_file = None, 
                 allow_speech_tags = allow_speech_tags, 
                 delimiters = sentence_delimiters):
        """
        Keyword arguments:
        stop_words_file  --  str，指定停止词文件路径（一行一个停止词），若为其他类型，则使用默认停止词文件
        delimiters       --  默认值是`?!;？！。；…\n`，用来将文本拆分为句子。
        
        Object Var:
        self.words_no_filter      --  对sentences中每个句子分词而得到的两级列表。
        self.words_no_stop_words  --  去掉words_no_filter中的停止词而得到的两级列表。
        self.words_all_filters    --  保留words_no_stop_words中指定词性的单词而得到的两级列表。
        """
        self.text = ''
        self.keywords = None
        
        self.seg = Segmentation(stop_words_file=stop_words_file, 
                                allow_speech_tags=allow_speech_tags, 
                                delimiters=delimiters)

        self.sentences = None
        self.words_no_filter = None     # 2维列表
        self.words_no_stop_words = None
        self.words_all_filters = None
        
    def analyze(self, text, 
                window = 2, 
                lower = False,
                vertex_source = 'all_filters',
                edge_source = 'no_stop_words',
                pagerank_config = {'alpha': 0.85,}):
        """分析文本

        Keyword arguments:
        text       --  文本内容，字符串。
        window     --  窗口大小，int，用来构造单词之间的边。默认值为2。
        lower      --  是否将文本转换为小写。默认为False。
        vertex_source   --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点。
                            默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。关键词也来自`vertex_source`。
        edge_source     --  选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边。
                            默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数。
        """
        
        # self.text = util.as_text(text)
        self.text = text
        self.word_index = {}
        self.index_word = {}
        self.keywords = []
        self.graph = None
        
        result = self.seg.segment(text=text, lower=lower)
        self.sentences = result.sentences
        self.words_no_filter = result.words_no_filter
        self.words_no_stop_words = result.words_no_stop_words
        self.words_all_filters   = result.words_all_filters

        debug(20*'*')
        debug('self.sentences in TextRank4Keyword:\n', ' || '.join(self.sentences))
        debug('self.words_no_filter in TextRank4Keyword:\n', self.words_no_filter)
        debug('self.words_no_stop_words in TextRank4Keyword:\n', self.words_no_stop_words)
        debug('self.words_all_filters in TextRank4Keyword:\n', self.words_all_filters)


        options = ['no_filter', 'no_stop_words', 'all_filters']

        if vertex_source in options:
            _vertex_source = result['words_'+vertex_source]
        else:
            _vertex_source = result['words_all_filters']

        if edge_source in options:
            _edge_source   = result['words_'+edge_source]
        else:
            _edge_source   = result['words_no_stop_words']

        self.keywords = sort_words(_vertex_source, _edge_source, window = window, pagerank_config = pagerank_config)

    def get_keywords(self, num = 6, word_min_len = 1):
        """获取最重要的num个长度大于等于word_min_len的关键词。

        Return:
        关键词列表。
        """
        result = []
        count = 0
        for item in self.keywords:
            if count >= num:
                break
            if len(item.word) >= word_min_len:
                result.append(item)
                count += 1
        return result
    
    def get_keyphrases(self, keywords_num = 12, min_occur_num = 2): 
        """获取关键短语。
        获取 keywords_num 个关键词构造的可能出现的短语，要求这个短语在原文本中至少出现的次数为min_occur_num。

        Return:
        关键短语的列表。
        """
        keywords_set = set([ item.word for item in self.get_keywords(num=keywords_num, word_min_len = 1)])
        keyphrases = set()
        for sentence in self.words_no_filter:
            one = []
            for word in sentence:
                if word in keywords_set:
                    one.append(word)
                else:
                    if len(one) >  1:
                        keyphrases.add(''.join(one))
                    if len(one) == 0:
                        continue
                    else:
                        one = []
            # 兜底
            if len(one) >  1:
                keyphrases.add(''.join(one))

        return [phrase for phrase in keyphrases 
                if self.text.count(phrase) >= min_occur_num]
###############################
















##########################################################################################################################
def quiet_logs( sc ):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
  logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )


def is_good(w):
    if re.findall(u'[\u4e00-\u9fa5]', w) \
        and len(w) >= 2\
        and not re.findall(u'[较很越增]|[多少大小长短高低好差]', w)\
        and not u'加一单' in w\
        and not u'加一款' in w\
        and not u'谢谢' in w\
        and not u'微信' in w\
        and not u'什么' in w\
        and not u'标准' in w\
        and not u'地方' in w\
        and not u'热带雨林' in w\
        and not u'一笑而过' in w\
        and not u'普罗旺斯' in w\
        and not (u'老板娘' in w or u'十里桃花' in w or u'一路有你' in w or u'朋友圈' in w)\
        and not w[-1] in u'课接收湖江间会社洋节为一人给内中后省市局院上所在有与及厂稿下厅部商者从奖出站总路街弄巷口田山区县店馆城个条块单欠班钱室我你他吧吗啦啊乐呢港苑村您呀'\
        and not w[0] in u'在发带未代要买已再支把袋斤两组台箱打颗根张粒朵欠班支把日号月年转请第每各该个被其从与及当为祝给不买卖借送收先我你他'\
        and not w[-2:] in [ u'过户', u'广场',u'回来',u'妈妈',u'爸爸',u'上午',u'下午',u'晚上',u'可以', u'季度',u'问题', u'市场', u'邮件', u'合约', u'假设', u'编号', u'预算', u'施加', u'战略', u'状况', u'工作', u'考核', u'评估', u'需求', u'沟通', u'阶段', u'账号', u'意识', u'价值', u'事故', u'竞争', u'交易', u'趋势', u'主任', u'价格', u'门户', u'治区', u'培养', u'职责', u'社会', u'主义', u'办法', u'干部', u'员会', u'商务', u'发展', u'原因', u'情况', u'国家', u'园区', u'伙伴', u'对手', u'目标', u'委员', u'人员', u'如下', u'况下', u'见图', u'全国', u'创新', u'共享', u'资讯', u'队伍', u'农村', u'贡献', u'争力', u'地区', u'客户', u'领域', u'查询', u'应用', u'可以', u'运营', u'成员', u'书记', u'附近', u'结果', u'经理', u'学位', u'经营', u'思想', u'监管', u'能力', u'责任', u'意见', u'精神', u'讲话', u'营销', u'业务', u'总裁', u'见表', u'电力', u'主编', u'作者', u'专辑', u'学报', u'创建', u'支持', u'资助', u'规划', u'计划', u'资金', u'代表', u'部门', u'版社', u'表明', u'证明', u'专家', u'教授', u'教师', u'基金', u'如图', u'位于', u'从事', u'公司', u'企业', u'专业', u'思路', u'集团', u'建设', u'管理', u'水平', u'领导', u'体系', u'政务', u'单位', u'部分', u'董事', u'院士', u'经济', u'意义', u'内部', u'项目', u'建设', u'服务', u'总部', u'管理', u'讨论', u'改进', u'文献']\
        and not w[:2] in [u'帮忙',u'有就',u'还是',u'上午',u'下午',u'分开',u'可以',u'晚上',u'昨天', u'今天',u'明天',u'爱你', u'考虑', u'图中', u'每个', u'出席', u'一个', u'随着', u'不会', u'本次', u'产生', u'查询', u'是否', u'作者']\
        and not (u'博士' in w or u'硕士' in w or u'研究生' in w)\
        and not (len(set(w)) == 1 and len(w) > 1)\
        and not (w[0] in u'毕索嘉宏雷齐付阿余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段漕钱汤尹黎易常武乔贺赖龚文段覃尹房崔季李王张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎' and len(w) == 3)\
        and not (w[-1] in u'妈爸' and len(w) == 3)\
        and not (w[0] in u'李王张刘陈杨赵黄周吴' and len(w) == 2)\
        and not (w[0] in u'一二三四五六七八九十' and len(w) == 2)\
        and not (w[-2:] in [u'师傅',u'老板',u'小姐',u'老师',u'女士',u'先生',u'年级',u'张卡',u'一个',u'道口'] and len(w) == 3)\
        and not ((u'节哀' in w or u'真言' in w or u'辛苦' in w or u'运转' in w or u'财源' in w or u'风顺' in w or u'平安' in w or u'愉快' in w or u'感谢' in w or u'感恩' in w or u'开心' in w or u'顺利' in w or u'幸福' in w or u'阖家' in w or u'欢乐' in w or u'福如' in w or u'寿比' in w or u'腾达' in w or u'平步' in w or u'高升' in w or u'亨通' in w) and len(w) == 4)\
        and re.findall(u'[^一七厂月二夕气产兰丫田洲户尹尸甲乙日卜几口工旧门目曰石闷匕勺]', w)\
        and not u'进一步' in w:
        return True
    else:
        return False


def clean_text(l):
    l= re.sub(u'[琴前后这那下上半一二三四五六七八九十甘百千佰仟贰万两叁仨几][束码人头线门块条磅股期注吨起次个支把袋斤两组台箱打颗根张粒朵盒副包瓶双对月年本套顶份套篮本罐桶件床只枚盆扎碗十百千万分季天声升日号]',' ', l)
    l=re.sub(u'[\u4e00-\u9fa5][束码人头线门块条股期注吨起次个支把袋斤两组台箱打颗根张磅粒朵盒副包瓶双对月本套顶份套篮本罐桶件床只枚盆扎碗十百千万分季天声升日号]',' ', l)
    l=re.sub(u'[0-9][束码人头线门块条股期注吨起次个支把袋斤两组台箱打颗根张粒朵盒副包瓶双对月本套顶份套篮本罐桶件床只枚盆扎碗十百千万分季天声升日号]',' ', l)
    l = re.sub(u'[的了这那到]', ' ', l)
    l = re.sub(u'[老清][帐婆公账]', ' ', l)
    l = re.sub(u'[我学会][会说]', ' ', l)
    l = re.sub(u'[是让请还帮给自][你我己要您]', ' ', l)
    l = re.sub(u'[不跟行][够发前是对行不]', ' ', l)
    l = re.sub(u'[补未已己][清上付扣][帐面款除次账]', ' ', l)
    l = re.sub(u'[补未已己付][清上付扣结]', ' ', l)
    l = re.sub(u'[亲宝][爱贝][的滴儿]', ' ', l)
    l = re.sub(u'[今明昨上下晌当][天日午年早晚次]', ' ', l)
    l = re.sub(u'[相看记还][信错得有]', ' ', l)
    l = re.sub(u'[支][付][宝]', ' ', l)
    l = re.sub(u'[星][期][一二三四五六七八日]', ' ', l)
    l = re.sub(u'[周][一二三四五六七八日]', ' ', l)
    l = re.sub(u'[感][谢恩]', ' ', l)
    l = re.sub(u'[供排][养列]', ' ', l)
    l = re.sub(u'[参工][加资]', ' ', l)
    l = re.sub(u'[这怎那好][么]', ' ', l)
    l = re.sub(u'[让进排色][分三球五]', ' ', l)
    l = re.sub(u'[五六][和合][彩]', ' ', l)
    l = re.sub(u'[租拿][金去]', ' ', l)
    l = re.sub(u'[都][是行可要]', ' ', l)
    l = re.sub(u'[主][持]', ' ', l)
    l = re.sub(u'[上下][楼车课马班脑样]', ' ', l)
    l = re.sub(u'[有无][限]', ' ', l)
    l = re.sub(u'[居果依悠虽][然]', ' ', l)
    l = re.sub(u'[^\u4e00-\u9fa5]+', ' ', l)
    return l


def get_ws_trans_text(tdw, table_name,partition_name):
    
    #df = tdw.table(tblName = table_name)
    
    #rdd = df.select('text').rdd.map(lambda row : row[0])
    
    #rdd = sc.textFile('./data/ws_wordsbase_0426V4.txt')

    df = tdw.table(tblName = table_name,priParts=[partition_name])
    
    #rdd = df..rdd.map(lambda row : row[0])



    return df[df['match_words_cnt']>5].select('txt_7days').rdd.map(lambda row : row[0]).map(lambda s : clean_text(s)).cache()


def get_other_trans_text(tdw, table_name):
    
    #df = tdw.table(tblName = table_name)
    
    #rdd = df.select('text').rdd.map(lambda row : row[0])
    
    #rdd = sc.textFile('./data/ws_wordsbase_0426V4.txt')

    df = tdw.table(tblName = table_name)
    
    #rdd = df..rdd.map(lambda row : row[0])



    return df.select('text').rdd.map(lambda row : row[0]).map(lambda s : clean_text(s)).cache()




# def get_other_trans_text(tdw, table_name):
    
#     #df = tdw.table(tblName = table_name)
    
#     #rdd = df.select('text').rdd.map(lambda row : row[0])
    
#     #rdd = sc.textFile('./data/ws_otherwordsbase_0426.txt')
#     df = sqlContext.read.csv('./data/data2.csv')

#     return df.rdd.map(lambda row : row[0]).map(lambda s : clean_text(s)).cache()



def get_city_list(tdw, table_name):
    
    #df = tdw.table(tblName = table_name)
    
    #rdd = df.select('text').rdd.map(lambda row : row[0])
    
    #rdd = sc.textFile('./data/citylist.csv')
    #df = sqlContext.read.csv('./data/citylist.csv')
    df = tdw.table(tblName = table_name)

    return df.select('key').rdd


def get_current_vocabulary_df(tdw, table_name):
    
    #df = tdw.table(tblName = table_name)
    
    #rdd = df.select('text').rdd.map(lambda row : row[0])
    
    #rdd = sc.textFile('./data/citylist.csv')
    #df = sqlContext.read.csv('./data/vcb_0429.csv', header=True)
    #clear_data_udf=udf(lambda z: clean_text(z),ArrayType(StringType()))
    df = tdw.table(tblName = table_name)
    #return df.select('key').rdd.map(lambda row : row[0]).map(lambda s : clean_text(s)).toDF(['key'])
    return df

def get_ground_truth_df(tdw, table_name):
    
    #df = tdw.table(tblName = table_name)
    
    #rdd = df.select('text').rdd.map(lambda row : row[0])
    
    #rdd = sc.textFile('./data/citylist.csv')
    #df = sqlContext.read.csv('./data/vcb_0429.csv', header=True)
    #clear_data_udf=udf(lambda z: clean_text(z),ArrayType(StringType()))
    df = tdw.table(tblName = table_name)
    #return df.select('key').rdd.map(lambda row : row[0]).map(lambda s : clean_text(s)).toDF(['key'])
    return df
        
def find_words(start_words, w2v_model, w2v_dict, center_words=None, neg_words=None, min_sim=0.6, max_sim=1., alpha=0.25):

    def most_similar(word, center_vec=None, neg_vec=None):
        vec=w2v_dict[word] + center_vec - neg_vec
        print(vec)
        return w2v_model.findSynonymsArray(vec, 200)
    if center_words == None and neg_words == None:
        min_sim = max(min_sim, 0.6)
    center_vec, neg_vec = np.zeros([word_size]), np.zeros([word_size])
    if center_words: # 中心向量是所有种子词向量的平均
        _ = 0
        for w in center_words:
            if w in w2v_model.wv.vocab:
                center_vec += w2v_dict[w]
                _ += 1
        if _ > 0:
            center_vec /= _
    if neg_words: 
        _ = 0
        for w in neg_words:
            if w in w2v_model.wv.vocab:
                neg_vec += w2v_dict[w]
                _ += 1
        if _ > 0:
            neg_vec /= _
    queue_count = 1
    task_count = 0
    cluster = []
    queue = Queue() # 建立队列
    for w in start_words:
        queue.put((0, w))
        if w not in cluster:
            cluster.append(w)
    while not queue.empty():
        idx, word = queue.get()
        #print(word)
        #print(center_vec)
        #print(neg_vec)
        #print(w2v_dict[word])
        queue_count -= 1
        task_count += 1
        sims = most_similar(word, center_vec, neg_vec)
        min_sim_ = min_sim + (max_sim-min_sim) * (1-np.exp(-alpha*idx))
        if task_count % 10 == 0:
            log = '%s in cluster, %s in queue, %s tasks done, %s min_sim'%(len(cluster), queue_count, task_count, min_sim_)
            print (log)
        for i,j in sims:
            if j >= min_sim_:
                if i not in cluster and is_good(i): # is_good是人工写的过滤规则
                    queue.put((idx+1, i))
                    if i not in cluster and is_good(i):
                        cluster.append(i)
                    queue_count += 1
    return cluster

def textrank_compute(words,other_words):
    #from textrank4zh import TextRank4Keyword
    print('import textrank4zh')
    

    #text = codecs.open('./doc/02.txt', 'r', 'utf-8').read()
    #text = "世界的美好。世界美国英国。 世界和平。"
    #text = codecs.open('../data/ws_wordsbase_0426V4.txt', 'r', 'utf-8').read()

    #Start TextRank Model

    size1=min(int(words.count()/4),1000)
    size2=min(int(other_words.count()/8),500)
    if size2>size1:
        size2=size1



    #text = codecs.open('./doc/02.txt', 'r', 'utf-8').read()
    #text = "世界的美好。世界美国英国。 世界和平。"

    tr4w = TextRank4Keyword()
    for i in words.collect():
        tr4w.analyze(text=i,lower=True, window=2, pagerank_config={'alpha':0.85})

    #text = codecs.open('./doc/02.txt', 'r', 'utf-8').read()
    #text = "世界的美好。世界美国英国。 世界和平。"

    #tr4w_result=tr4w.get_keywords(20000, word_min_len=2)

    #list1=[]
    list2=[]
    for item in tr4w.get_keywords(size1, word_min_len=2):
    #   list1.append([item.word, item.weight])
        list2.append(item.word)
    #dictionary1=tr4w.get_keywords(20000, word_min_len=2)
    rdd_result1 = sc.parallelize(list2)
    
    #rdd_result1.toDF(['words','values']).write.csv('mycsv.csv')

    #import pandas as pd
    #import numpy as np
    #pd.DataFrame(list1).to_csv(datapath+'TextRank_result3.csv', encoding='utf-8-sig', header=None, index=None)
    print('Output TextRank Result!')
    #list2_array=np.array(list1)
    #list2_series=pd.Series(list2_array[:,1])
    #list2_series.index=list2_array[:,0]
    #list2_series.to_csv(datapath+'TextRank_result_ws3_0426.csv', encoding='utf-8-sig', header=None, index=None)
    #save_feature_tdw(tdw, feature_df, feature_out_table) #Output the data to Output Data TDW Table, which is defined in sys.argv[8]
 
    

    # =============================================================================
    # from textrank4zh import TextRank4Sentence
    # 
    # #text = codecs.open('./doc/03.txt', 'r', 'utf-8').read()
    # text = "这间酒店位于北京东三环，里面摆放很多雕塑，文艺气息十足。答谢宴于晚上8点开始。"
    # tr4s = TextRank4Sentence()
    # tr4s.analyze(text=text, lower=True, source = 'all_filters')
    # 
    # for st in tr4s.sentences:
    #     print(type(st), st)
    # 
    # print(20*'*')
    # for item in tr4s.get_key_sentences(num=4):
    #     print(item.weight, item.sentence, type(item.sentence))
    # =============================================================================

    #text2 = codecs.open('../data/ws_otherwordsbase_0426.txt', 'r', 'utf-8').read()
    tr4w_v2 = TextRank4Keyword()
    for i in other_words.collect():
        tr4w_v2.analyze(text=i,lower=True, window=2, pagerank_config={'alpha':0.85})
    #text = codecs.open('./doc/02.txt', 'r', 'utf-8').read()
    #text = "世界的美好。世界美国英国。 世界和平。"

    #olist1=[]
    olist2=[]
    for item in tr4w_v2.get_keywords(size2, word_min_len=2):
    #    olist1.append([item.word, item.weight])
        olist2.append(item.word)
    #dictionary2 = tr4w.get_keywords(20000, word_min_len=2)
    #rdd_result1 = sc.parallelize([dictionary1[0].keys()])
    rdd_result2 = sc.parallelize(olist2)
    #print(rdd_result2.collect())    

    #pd.DataFrame(list1).to_csv(datapath+'TextRank_result.csv', encoding='utf-8-sig', header=None, index=None)
    print('Output TextRank Result 2!')
    #olist2_array=np.array(list1)
    #olist2_series=pd.Series(olist2_array[:,1])
    #olist2_series.index=olist2_array[:,0]
    #olist2_series.to_csv(datapath+'TextRank_result_other3_0426.csv', encoding='utf-8-sig', header=None, index=None)


    #pd.DataFrame(list(set(list2)-set(olist2))).to_csv(datapath+'TextRank_result_final3_0426.csv', encoding='utf-8-sig', header=None, index=None)
    print('Finish!!!')
    return rdd_result1.subtract(rdd_result2)


def compute(rdd_words,rdd_other_words, df_vocabulary,rdd_city,window_size,word_size,min_count_size,threshold1,threshold2,power):
    
    
    def D(iter):
        for i in iter.collect():
            yield i


    
    f = Word_Finder(min_proba=1e-6, min_pmi=0.5)
    f.train(D(rdd_words)) 
    f.find(D(rdd_words)) 


    #rdd_words = sc.parallelize([f.words])
    #print(rdd_words.collect())
    words = pd.Series(f.words).sort_values(ascending=False)

    #print(rdd_words.values().sum())
    print('Find the ws words!')


    
    fo = Word_Finder(min_proba=1e-6, min_pmi=0.5)
    fo.train(D(rdd_other_words)) 
    fo.find(D(rdd_other_words)) 
    







    other_words = pd.Series(fo.words).sort_values(ascending=False)

    #rdd_otherwords = sc.parallelize([fo.words])
    #print(rdd_otherwords.values().sum())
    print('Find the other words!')


    other_words = other_words / other_words.sum() * words.sum() # 总词频归一化




    WORDS = words.copy()
    OTHER_WORDS = other_words.copy()




    total_zeros = (WORDS + OTHER_WORDS).fillna(0) * 0
    words = WORDS + total_zeros
    other_words = OTHER_WORDS + total_zeros
    total = words + other_words

    alpha = words.sum() / total.sum()

    result = (words + total.mean() * alpha) / (total + total.mean())
    result = result.sort_values(ascending=False)
    idxs = [i for i in result.index if len(i) >= 2 and is_good(i)]
     # 排除掉单字词
    idxs15000=idxs[:15000]
    print('Finish Normalize!')

    print('The length of idxs is'+str(len(idxs)))




    print('Output PMI Result!')


    rdd_textrank = textrank_compute(rdd_words,rdd_other_words)
    rdd_textrank2=rdd_textrank.filter(lambda x: len(x)>=2 and is_good(x))
    print('The textrank length is '+str(rdd_textrank2.count()))


    #rdd_city = get_city_list(tdw,tdw_city)



    import jieba
    print('Import Jieba!!!!!!')
    #jieba.load_userdict("./dic.txt")
    for i in idxs:
        if words[i]>0:
            jieba.add_word(i)
        else:
            jieba.add_word(i)
            #jieba.suggest_freq(i,True)

    for item in rdd_textrank2.collect():
        jieba.add_word(item)
        #jieba.suggest_freq(item,True)
        
    for city in [row['key'] for row in rdd_city.collect()]:
        jieba.add_word(city)
        #jieba.suggest_freq(city,True)

        
    #tokenizer = jieba.lcut


    #text = codecs.open('./doc/02.txt', 'r', 'utf-8').read()
    #text = "世界的美好。世界美国英国。 世界和平。"
    #for item in tr4w.get_keywords(30, word_min_len=2):
    #    print(item.word, item.weight, type(item.word))
    #tokenizer=jieba.Tokenizer
    # =============================================================================
    #tokenizer = f.export_tokenizer()
    #class DW:
    #    def __iter__(self):
    #        for l in D(rdd_words):
    #            yield tokenizer.tokenize(l, combine_Aa123=True)

    jieba_result=rdd_words.map(lambda x: list(jieba.cut(x))).filter(lambda x: x!=' ')
    print('jieba cut  done!!!!!!!')         

    
    #from gensim.models import Word2Vec

    #word2vec = Word2Vec(jieba_result.collect(), size=word_size, window=20, min_count=2, sg=1, negative=10)
    

    from pyspark.ml.feature import Word2Vec
    word2Vec = Word2Vec(vectorSize=word_size,  minCount=min_count_size, windowSize=window_size, inputCol="text", outputCol="result")
    
    print('Define the Word2vec Model')
    spark= SparkSession\
                .builder \
                .appName("dataFrame") \
                .getOrCreate()

    # Input data: Each row is a bag of words from a sentence or document.

    documentDF5 = spark.createDataFrame(jieba_result.map(lambda x: (x,)), ["text"])
    documentDF5.show(5)
    #documentDF6 = spark.createDataFrame([rdd_words.map(row)], ["text"])
    #documentDF6.show(3)       
    #documentDF4 = spark.createDataFrame(rdd_words.map(lambda x: x.split(" ")), ["text"])
    #documentDF4.show(3)
    #documentDF3 = spark.createDataFrame((rdd_words.map(lambda x: x.split(" ")),), ["text"])
    #documentDF3.show(3)
    #documentDF = spark.createDataFrame([(rdd_words.map(lambda x: x.split(" ")),),(rdd_words.map(lambda x: x.split(" ")),)], ["text"])
    #documentDF.show(3)
    #documentDF2 = spark.createDataFrame([(rdd_words.map(lambda x: x.split(" ")),)], ["text"])
    #documentDF2.show(3)
   
    
    #model = word2Vec.fit(rdd_words.map(lambda x: x.split(" ")).toDF(['text']))
    model = word2Vec.fit(documentDF5)
    print('Finish Word2Vec!')


    def find_words(start_words, w2v_model, w2v_dict, center_words=None, neg_words=None, min_sim=0.6, max_sim=1., alpha=0.25):
    #from pyspark.ml.linalg import DenseVector

        def most_similar(word, center_vec=None, neg_vec=None):
            #print(word)
            vec=w2v_dict[word] + center_vec - neg_vec
            #print(vec)

            #vec_apache=DenseVector(vec)
            return w2v_model.findSynonyms(vec, 200).collect()
        if center_words == None and neg_words == None:
            min_sim = max(min_sim, 0.6)
        center_vec, neg_vec = np.zeros([word_size]), np.zeros([word_size])
        if center_words: # 中心向量是所有种子词向量的平均
            _ = 0
            for w in center_words:
                if w in w2v_model.wv.vocab:
                    center_vec += w2v_dict[w]
                    _ += 1
            if _ > 0:
                center_vec /= _
        if neg_words: 
            _ = 0
            for w in neg_words:
                if w in w2v_model.wv.vocab:
                    neg_vec += w2v_dict[w]
                    _ += 1
            if _ > 0:
                neg_vec /= _
        queue_count = 1
        task_count = 0
        #cluster = []
        cluster_dict=dict()
        queue = Queue() # 建立队列
        for w in start_words:
            queue.put((0, w))
            if w not in cluster_dict.keys():
                cluster_dict[w]=1
        while not queue.empty():
            idx, word = queue.get()
                #print(word)
                #print(center_vec)
                #print(neg_vec)
                #print(w2v_dict[word])
            queue_count -= 1
            task_count += 1
            sims = most_similar(word, center_vec, neg_vec)
            min_sim_ = min_sim + (max_sim-min_sim) * (1-np.exp(-alpha*idx))
            #if task_count % 10 == 0:
            #    log = '%s in cluster, %s in queue, %s tasks done, %s min_sim'%(len(cluster_dict), queue_count, task_count, min_sim_)
            #    print (log)
            for i,j in sims:
                if j >= min_sim_:
                    if i not in cluster_dict.keys() and is_good(i): # is_good是人工写的过滤规则
                        queue.put((idx+1, i))
                            #if i not in cluster and is_good(i):
                        cluster_dict[i]=cluster_dict[word]*j
                            #cluster.append(i)
                        queue_count += 1
                    elif is_good(i):
                        if cluster_dict[word]*j>cluster_dict[i]:
                            cluster_dict[i]=cluster_dict[word]*j
        return_value={k: v for k, v in cluster_dict.items() if v>=min_sim}
        return return_value




    w2v_df=model.getVectors()
    w2v_df.show()
    ta=w2v_df.alias('ta')
    print('Finish Word2Vec!!!')

    #cur_word_list = [row['key'] for row in rdd_vocabulary.collect()]
    df_vocabulary.show(20)
    tb=df_vocabulary.alias('tb')
    start_words=tb.join(ta,tb.key==ta.word).select('word','vector')
    start_words.show()

    list_words = map(lambda row: row.asDict(), w2v_df.collect())
    dict_words = {item['word']:item['vector'] for item in list_words}
    
    print('The dict length is'+str(len(dict_words)))


    cluster_words = find_words(start_words.select('word').rdd.flatMap(lambda x: x).collect(), model,dict_words, min_sim=threshold1, alpha=threshold2)
    result_dict=result[cluster_words.keys()].to_dict()
    result_merge={k:result_dict[k]*cluster_words[k]**power for k in result_dict}
    result2 = pd.Series(result_merge).sort_values(ascending=False)
    

    def build_ac_tree(pattern_word_list):
        
        ac = ACT.ACTree()
        ac.build(pattern_word_list)
        
        return ac


    list_cur_vcb=df_vocabulary.select('key').rdd.flatMap(lambda x: x).collect()
    AC_Tree=build_ac_tree(list_cur_vcb)
    AC_Tree_City=build_ac_tree(rdd_city.collect())

    def is_not_cover(w):
        match_res = AC_Tree.match(w)
        words_set=set(match_res)
        if not words_set:
            return True
        else:
            return False
        
    def is_city(w):
        match_res = AC_Tree_City.match(w)
        words_set=set(match_res)
        if not words_set:
            return True
        else:
            return False

    def is_rare_name(string):
        pattern = re.compile(u"[~!@#$%^&* ]")
        match = pattern.search(string)
        if match:
            return False
        try:
            string.encode("gb2312")
        except UnicodeEncodeError:
            return False
        return True




    idxs = [i for i in result2.index if is_good(i) and is_not_cover(i) and is_rare_name(i) and is_city(i)]
    idxs_v=[]
    for i in idxs:
        for j in list_cur_vcb:
            flag=0
            if re.match(i,j):
                flag=1
                break
        if flag==0:
            idxs_v.append(i)


    return sc.parallelize([i for i in idxs_v if len(i) >= 2])

def save_feature_tdw(tdw, feature_df, table_name):
    '''
    save feature to local file
    '''

    tdw.saveToTable(feature_df, tblName = table_name)


def save_feature_tdw_partition(tdw, feature_df, table_name,partition_name):
    '''
    save feature to local file
    '''

    tdw.saveToTable(feature_df, tblName = table_name,priPart=partition_name)

def create_partition(tdwUtil, tablename, pname, part):
    if not tdwUtil.partitionExist(tablename, pname):
        tdwUtil.createListPartition(tablename, pname, part)

def run():
    
    if len(sys.argv) != 13:
        print(len(sys.argv))
        sys.stderr.write("usage: %s [i]ws_text_table [i]other_text_table [i]citylist_table [i]cur_vcb_table [i]feature_out_table\n")
        sys.exit(1)
        
        
    ws_text_table = sys.argv[1]  #Input Data TDW Table
    other_text_table = sys.argv[2]  #Stopwords TDW Table
    citylist_table = sys.argv[3] # Output Data Mid TDW Table
    cur_vcb_table = sys.argv[4]
    #max_ngram_len = int(sys.argv[4]) #Parameter1
    #min_word_cnt = int(sys.argv[5]) #Parameter2
    #min_firmness = float(sys.argv[6]) #Parameter3
    #min_dof = float(sys.argv[7]) #Parameter4
    ground_truth_table=sys.argv[5]
    feature_out_table = sys.argv[6]  #Output Data TDW Table
    window_size=int(sys.argv[7])
    word_size=int(sys.argv[8])
    min_count_size=int(sys.argv[9])
    T1=float(sys.argv[10])
    T2=float(sys.argv[11])
    power=float(sys.argv[12])


    tdw = TDWSQLProvider(session, user="tdw_yorksywang", passwd="bq0602BQB", db="fkana_db", group = 'cft') #Set User Id and password
    tdwUtil = TDWUtil(user="tdw_yorksywang", passwd="bq0602BQB", dbName="fkana_db", group = 'cft') #Set User Id and password
    
    #table_schema = get_feature_table_schema() #Hardcode, the desciprtion of the output table
    #create_table(tdwUtil, feature_out_table, table_schema)#Create the ouput table with the hardcode description
    #spark=SparkSession(sc)
    print('Start Reading Weishang')
    
    date = time.strftime('%Y%m%d', time.localtime(time.time()-86400 ))
    partition_name = 'p_'+date
    print('fpar_date is '+partition_name)
    rdd_ws_trans_text = get_ws_trans_text(tdw, ws_text_table,partition_name) #Get the data for NLP process
    #df=spark.createDataFrame(['words'])
    #df.write.csv("sqlcsvA.csv")
    #print(rdd_ws_trans_text.collect())
    print('Start Reading Gamble')
    rdd_other_trans_text=get_other_trans_text(tdw,other_text_table)
    
    print('Start Reading Current Vocabulary')
    #vcb_rdd=get_current_vocabulary_df(tdw,cur_vcb_table)

    print('Start Reading City_Name_Relationship list')
    city_rdd=get_city_list(tdw, citylist_table)
    print('The city name list has '+str(city_rdd.count()))

    print('Start Getting the Ground Truth list')
    #truth_rdd=get_ground_truth_df(tdw,ground_truth_table).map(lambda x: x['key'])
    truth_rdd=get_ground_truth_df(tdw,ground_truth_table)
    #stop_words_set = get_stop_words_set(tdw, stop_words_table)
    #discovered_words_set = get_discovered_words_set(tdw, discovered_words_table)#Current Vocabulary
    print('Start Vocabulary Mining')
    mining_vocabulary=compute(rdd_ws_trans_text,rdd_other_trans_text,truth_rdd,city_rdd,window_size,word_size,min_count_size,T1,T2,power)
    
    #print(mining_vocabulary.take(100))
    #print(truth_rdd.take(100))

    print('The total size is '+str(mining_vocabulary.count()))
    #textrank_keywords.map(row).toDF(['words']).write.option("encoding", "UTF-8").csv(path=file, header=True, sep=",", mode='overwrite')
    print('Finish Writing')
    #feature_df = compute(session, rdd_trans_text, stop_words_set, discovered_words_set, max_ngram_len, min_word_cnt, min_firmness, min_dof)#NLP Process
    
    #save_feature_tdw(tdw, feature_df, feature_out_table) #Output the data to Output Data TDW Table, which is defined in sys.argv[8]
    
    #save_feature_tdw(tdw, session.createDataFrame(sc.parallelize(mining_vocabulary.take(1000)), StringType()), feature_out_table)
    #save_feature_tdw_partition(tdw, session.createDataFrame(sc.parallelize(mining_vocabulary.take(100)), StringType()), feature_out_table,partition_name)
    output_schema=StructType()
    output_schema.add(StructField(name = 'fpar_date', dataType=StringType())) 
    output_schema.add(StructField(name = 'fbuy_qq', dataType = StringType()))
    create_partition(tdwUtil,feature_out_table,partition_name,date)
    save_feature_tdw_partition(tdw, session.createDataFrame(sc.parallelize(mining_vocabulary.take(100)).map(lambda x: [date,x]), output_schema), feature_out_table,partition_name)
#spark.stop()
    
    
    
if __name__ == "__main__":
    sc =SparkContext.getOrCreate()
    #quiet_logs(sc)
    session = SparkSession.builder.appName("trans_text_feature_compute").getOrCreate() #Create app with name
    sqlContext = SQLContext(sc)
    row = Row("val")
    #word_size = 32
    run()
    session.stop()
    print('END')