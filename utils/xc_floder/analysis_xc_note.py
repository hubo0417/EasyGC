import csv
import os
from threading import RLock
from utils.xc_floder.xc_spider import XC_Trip_City_Page_And_Note_Spider
from utils.xc_floder.config import CSV_CITY_NOTES_PATH
from embeddings.embedding_helper import EmbeddingHelper


class Analysis_XC_Note:

    sort_field: str = "点赞数"
    is_use_orginal_Image: bool = False  # 是否直接使用原始图片
    top: int = 2
    single_lock = RLock()

    def __init__(self, **kwargs) -> None:
        if "sort_field" in kwargs:
            self.sort_field = kwargs["sort_field"]
        if "is_use_orginal_Image" in kwargs:
            self.is_use_orginal_Image = bool(kwargs["is_use_orginal_Image"])
        if "top" in kwargs:
            self.top = int(kwargs["top"])

    @staticmethod
    def read_note_list(city: str,
                       top: int = 1,
                       is_from_embedding_db: bool = False):
        if is_from_embedding_db is False:
            datas = []
            file_path = CSV_CITY_NOTES_PATH.format(city)
            # 如果没有对应文件，则开启爬虫，生成对应城市文件
            if os.path.isfile(file_path) is False:
                with Analysis_XC_Note.single_lock:
                    if os.path.isfile(file_path) is False:
                        city_url = XC_Trip_City_Page_And_Note_Spider.get_city_base_url(
                            city=city)
                        spider = XC_Trip_City_Page_And_Note_Spider()
                        spider.get_city_url_and_notes(city_url, "", city=city)
            if os.path.isfile(file_path) is False:
                raise ValueError("未能找到对应文件")
            with open(file_path, 'r', newline='',
                      encoding='utf-8-sig') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    datas.append(row)
        else:
            helper = EmbeddingHelper(collection_name="travel")
            datas = helper.query(city,
                                 10,
                                 is_find_metedata=True,
                                 filter={"source": "xc"},
                                 where_document={"$contains": city})
        if datas and len(datas) > 0:
            filtered_data = [item for item in datas if item["图片集合"] != "[]"]
            sort_result = sorted(
                filtered_data,
                key=lambda x: int(x[Analysis_XC_Note.sort_field]),
                reverse=True)[:top]
            return sort_result
        else:
            return None
