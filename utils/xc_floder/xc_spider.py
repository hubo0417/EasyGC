import os
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
import re
import random
import time
import csv
from utils.xc_floder.config import CSV_CITY_NOTES_PATH, CSV_CITY_LINK_PATH, XC_COOKIE
from embeddings.embedding_helper import EmbeddingHelper
import json


class XC_Trip_City_Link_Spider:
    """
    初始化写入全球各地城市对应的URL到CSV文件
    """

    url = "https://you.ctrip.com/"
    headerS = {
        'user-agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 Edg/90.0.818.62'
    }

    @classmethod
    def init_city_link(cls):
        res = requests.get(url=cls.url, headers=cls.headerS)
        soup = BeautifulSoup(res.text, 'html.parser')
        city_list = soup.find_all(
            'a', class_="city-selector-tab-main-city-list-item")
        datas = []
        for item in city_list:
            chengshi = item.get_text()
            dizhi = item["href"]
            dizhi = dizhi.replace('place', 'travels')
            data = {'城市': chengshi, '地址': dizhi}
            datas.append(data)
        with open(CSV_CITY_LINK_PATH, 'w', newline='',
                  encoding='utf-8-sig') as f:
            writer = csv.DictWriter(
                f, fieldnames=['城市', '地址'])  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
            writer.writeheader()  # 写入列名
            writer.writerows(datas)  # 写入数据 travels

    @classmethod
    def read_city_info(cls):
        datas = []
        with open(CSV_CITY_LINK_PATH, 'r', newline='',
                  encoding='utf-8-sig') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                datas.append(row)
        return datas


class XC_Trip_City_Page_And_Note_Spider:
    """
    获取各个城市对应页面下的分页连接以及游记列表中的游记详情地址
    """
    # 请求头
    headerS = {
        'user-agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 Edg/90.0.818.62'
    }
    base_url = "https://you.ctrip.com"

    # 一次最多获取多少页记录
    max_epoch_page_count: int = 1

    def __init__(self, **kwargs) -> None:
        if "max_epoch_page_count" in kwargs:
            self.max_epoch_page_count = int(kwargs["max_epoch_page_count"])

    @classmethod
    def get_city_base_url(cls, city: str):
        if not city:
            raise ValueError("未能获取到城市名称")
        city_datas = XC_Trip_City_Link_Spider.read_city_info()
        for item in city_datas:
            if item[0] == city:
                city_url = item[1]
                break
        return city_url

    def get_city_url_and_notes(self,
                               url: str,
                               refer_url: str,
                               city: str,
                               iterations: int = 1,
                               data_list=[]):

        detail_helper = XC_Trip_Note_Detail_Spider()

        def get_note_details_and_next_page(list_url: str):
            # 获取第一页url
            res = requests.get(url=list_url, headers=self.headerS)
            soup = BeautifulSoup(res.text, 'html.parser')
            note_list = soup.find_all('a', class_="journal-item cf")
            for i in note_list:
                note_url = f"{self.base_url}{i['href']}"
                try:
                    data = detail_helper.get_xc_detail(note_url)
                    if data is not None:
                        data_list.append(data)
                except Exception:
                    pass
            try:
                next_page = soup.find_all("a", class_="nextpage")
                next_page_url = next_page[0]["href"]
            except Exception:
                next_page_url = None
            return next_page_url

        if not url:
            raise ValueError("未能得到对应页面的URL")
        if not refer_url:
            refer_url = url
        self.headerS["referer"] = refer_url
        # 根据当前页找到下一页的url地址
        next_page_url = get_note_details_and_next_page(url)
        try:
            while next_page_url is not None and iterations <= self.max_epoch_page_count:
                time.sleep(random.random() * 10)
                next_page_url = f"{self.base_url}{next_page_url}"
                next_page_url = get_note_details_and_next_page(next_page_url)
                iterations += 1
        except Exception:
            pass

        # 初始化CSV文件
        file, writer = detail_helper.init_detail_csv(city)
        writer.writerows(data_list)
        file.close()

        # 写入向量数据库
        self.embedding_notes(data_list)

    def embedding_notes(self, notes: list):

        embedding_helper = EmbeddingHelper(collection_name="travel")
        if notes and len(notes) > 0:
            for note in notes:
                data = note["正文"]
                texts = EmbeddingHelper.splitText(data,
                                                  chunk_size=1000,
                                                  overlap=0)
                if isinstance(note["图片集合"], list):
                    note["图片集合"] = json.dumps(note["图片集合"])
                note["source"] = "xc"
                embedding_helper.embedding_texts(texts=texts, metadatas=[note])


class XC_Trip_Note_Detail_Spider:
    # 基础爬取地址
    base_url = 'https://you.ctrip.com'
    # 请求头 伪装浏览器信息
    headers = {
        'user-agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 Edg/90.0.818.62',
    }
    csv_head = [
        '文章标题', '发布时间', '正文', '图片集合', '图片数量', '游玩天数', '游玩时间', '人均消费', '和谁',
        '玩法', '浏览量', '点赞数'
    ]
    cookie = XC_COOKIE
    # 评论分隔符
    SPLIT = 'y7qEE#Ri99'

    def init_detail_csv(self, city: str):
        file_path = CSV_CITY_NOTES_PATH.format(city)
        if os.path.isfile(file_path) is False:
            f = open(file_path, 'a', newline='', encoding='utf-8-sig')
            writer = csv.DictWriter(
                f, fieldnames=self.csv_head)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
            writer.writeheader()  # 写入列名
        else:
            f = open(file_path, 'a', newline='', encoding='utf-8-sig')
            writer = csv.DictWriter(f)
        return f, writer

    @classmethod
    def get_xc_detail(cls, url: str) -> dict:
        """
        :param url: 文章地址
        :return:
        """
        # 初始化数据，未获取则为 -1
        title, pub_time, img_count, tianshu, shijian, renjun, heshui, wanfa, VisitCount, LikeCount = [
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        ]

        def has_html_tags(input_string):
            # 使用正则表达式匹配HTML标签
            pattern = re.compile(r'<.*?>')
            return bool(pattern.search(input_string))

        # 获取游记互动信息
        def get_hudong(num: int):
            """
            获取互动信息
            :param num:文章id
            :return:
            """
            headers = {
                'user-agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 Edg/90.0.818.62',
            }
            url = 'https://you.ctrip.com/TravelSite/Home/GetBusinessData'
            params = {
                'random': random.random(),
                'arList[0].RetCode': 0,
                'arList[0].Html': num,
                'arList[1].RetCode': 1,
                'arList[1].Html': num,
                'arList[2].RetCode': 2,
                'arList[2].Html': num,
                'arList[3].RetCode': 3,
                'arList[3].Html': 158,
            }
            res = requests.get(url=url, headers=headers, params=params)
            # 获取数据所在str
            Html = dict(res.json()[1])['Html']
            # 初始化数据
            VisitCount, LikeCount = (-1, -1)
            try:
                # 浏览量
                VisitCount = re.findall('"VisitCount":(\d*?),', Html)[0]
            except Exception:
                pass
            # 点赞数 --> 页面为显示，意义未知
            try:
                LikeCount = re.findall('"LikeCount":(\d*?),', Html)[1]
            except Exception:
                pass
            return VisitCount, LikeCount

        # 获取游记文字内容
        def get_body(texts: str) -> str:
            # 去除图片
            x = re.sub(r'<p><div class="img".*?</a></div></p>', '', texts)
            # 去除<p><br/></p>
            y = re.sub(r'<p><br/></p>', '', x)
            # 去除<p><strong><br/></strong></p>
            z = re.sub(r'<p><strong><br/></strong></p>', '', y)
            # 去除 a标签
            a = re.sub(r'<a class="gs_a_poi.*?target="_blank">', '', z)
            a = re.sub(r'</a>|<a .*?>', '', a)
            # 去除其它富文本标签
            clear = re.sub(
                r'推荐住宿.*?\d{2}:\d{2}|发表于.*?\d{2}:\d{2}|<video.*?>.*?</video>|<p><strong>|</strong></p>|<p>|</p>|<br/>|<strong.*?>|</strong.*?>|\xa0|<span class="price">|<em>|</em>|<h\d class.*?>|</h\d>|<h\d>|<div class="img" data-likecategory="1".*?>|<div class="img_blk.*?</div>|</div>|<div .*?>|<img .*?>|<iframe .*?</iframe>',
                '', a)
            is_has_html = has_html_tags(clear)
            if is_has_html is True:
                clear = re.sub(r'<.*?>', '', clear)
                is_has_html = has_html_tags(clear)
            return clear if is_has_html is False else ""

        # 获取响应
        res = requests.get(url=url, headers=cls.headers)
        # 解析内容
        soup = BeautifulSoup(res.text, 'html.parser')

        # 获取标题
        try:
            title = soup.title.string.split(' - ')[0]
        except Exception:
            pass

        # 获取发布时间
        try:
            pub_time = soup.find_all('div', class_="time")[0].string
        except Exception:
            # pub_time = re.findall(r'发表.*?(\d{4}-\d{2}-\d{2})', res.text)[0]
            pass
        # 获取文章信息--》天数、时间、人均、和谁、玩法
        try:
            ctd_content_controls = soup.find_all(
                'div', class_='ctd_content_controls cf')[0]
            head = str(ctd_content_controls)
        except Exception:
            pass
        # 天数
        try:
            tianshu = re.findall('<span><i class="days"></i>天数：(.*?)</span>',
                                 head)[0]
        except Exception:
            pass

        # 时间
        try:
            shijian = re.findall('<span><i class="times"></i>时间：(.*?)</span>',
                                 head)[0]
        except Exception:
            pass

        # 人均
        try:
            renjun = re.findall('<span><i class="costs"></i>人均：(.*?)</span>',
                                head)[0]
        except Exception:
            pass

        # 和谁
        try:
            heshui = re.findall('<span><i class="whos"></i>和谁：(.*?)</span>',
                                head)[0]
        except Exception:
            pass

        # 玩法
        try:
            wanfa = re.findall('<span><i class="plays"></i>玩法：(.*?)</span>',
                               head)[0]
        except Exception:
            pass

        # 获取带有富文本标签的正文
        ctd_content = soup.find_all('div', class_='ctd_content')[0]

        try:
            ctd_content_controls = soup.find_all(
                'div', class_='ctd_content_controls cf')[0]
            # print(ctd_content_controls)
            ctd_content = str(ctd_content).replace(str(ctd_content_controls),
                                                   '')
            texts = re.sub(
                '<div class="ctd_content">.*?发表于.*?</h\d>|<div class="ctd_content">',
                '', str(ctd_content))
        except Exception:
            texts = re.sub(
                '<div class="ctd_content">.*?发表于.*?</h\d>|<div class="ctd_content">',
                '', str(ctd_content))

        # 处理富文本标签,正文内容
        body = get_body(texts)
        if body == "":
            return None
        # 统计图片数量
        try:
            imgs_url: List[Dict[str, str]] = []
            # 定位正文标签
            ctd_content = soup.find_all('div', class_='ctd_content')[0]
            # 获取图片标签
            imgs_div = ctd_content.find_all('div', class_="img")
            for item in imgs_div:
                try:
                    img_a = item.find_all('a')
                    if img_a and len(img_a) > 0:
                        image_src = img_a[0]["href"]
                    else:
                        image_src = None
                    desc_div = item.find_all('div', class_='description')
                    if desc_div and len(desc_div) > 0:
                        image_desc = desc_div[0].get_text()
                    else:
                        image_desc = None

                    image_k_v = {"src": image_src, "desc": image_desc}
                    if image_k_v["src"] is not None:
                        imgs_url.append(image_k_v)
                except Exception:
                    pass

            # 获取图片数量
            img_count = len(imgs_div)
        except Exception:
            img_count = 0
            imgs_url = []

        # 统计互动数量
        try:
            VisitCount, LikeCount = get_hudong(int(url.split('/')[-1][:-5]))
        except Exception:
            pass

        # 构建数据
        data = {
            '文章标题': title,
            '发布时间': pub_time,
            '正文': body,
            '图片集合': imgs_url,
            '图片数量': img_count,
            '游玩天数': tianshu,
            '游玩时间': shijian,
            '人均消费': renjun,
            '和谁': heshui,
            '玩法': wanfa,
            '浏览量': VisitCount,
            '点赞数': LikeCount
        }

        return data


if __name__ == '__main__':
    city_url = XC_Trip_City_Page_And_Note_Spider.get_city_base_url("西宁")
    XC_Trip_City_Page_And_Note_Spider().get_city_url_and_notes(url=city_url,
                                                               refer_url="",
                                                               city="西宁")
