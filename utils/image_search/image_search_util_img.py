import json
import os
import uuid
from typing import Union
import chromadb
from chromadb.config import Settings
import requests
from blip.blip_model import Blip_Model
from configs.base_config import GOOGLE_APIKEY, GOOGLE_SEARCH_ID, BASE_CONFIG
from PIL import Image
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import uuid
from scipy.spatial import distance

from utils.translation import Translation_Baidu


class Image_Search_Util_Img:
    search_count_from_google: int = 10
    search_count_from_embedding_db: int = 4
    max_count_from_google: int = 40
    collection_name = "images_source"
    destination: str = "D:\\EasyGC\\images_download\\images"
    cnn_model = None
    db_collection: chromadb.Collection = None
    cos_distance_threshold = 0.7
    page_index: int = 1
    blip_info: str = None

    def __init__(self, **kwargs) -> None:
        if "search_count_from_google" in kwargs:
            self.search_count_from_google = int(
                kwargs["search_count_from_google"])
        if "search_count_from_embedding_db" in kwargs:
            self.search_count_from_embedding_db = int(
                kwargs["search_count_from_embedding_db"])
        if "cos_distance_threshold" in kwargs:
            self.cos_distance_threshold = float(
                kwargs["cos_distance_threshold"])
        if "page_index" in kwargs:
            self.page_index = float(kwargs["page_index"])
        if "max_count_from_google" in kwargs:
            self.max_count_from_google = float(kwargs["max_count_from_google"])
        # 初始化卷积网络模型
        cnn_inner_model = VGG16(include_top=False)
        self.cnn_model = Model(
            inputs=cnn_inner_model.input,
            outputs=cnn_inner_model.get_layer('block5_conv2').output)
        # 初始化向量数据库
        db = chromadb.Client(
            settings=Settings(persist_directory=BASE_CONFIG["chromadb_path"],
                              is_persistent=True))
        self.db_collection = db.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"})

    # 提取图片文本摘要
    def _blip_image(self, image_url: Union[list, str], prefix: str = None):
        if not image_url:
            raise ValueError("未能获取到对应图片地址,此参数不能为空")
        blip_model = Blip_Model()
        result: list = []
        if isinstance(image_url, list):
            for item in image_url:
                blip_str = blip_model.generate_text_from_image(image_url=item,
                                                               text=prefix)
                result.append(blip_str)
        else:
            blip_str = blip_model.generate_text_from_image(image_url=image_url,
                                                           text=prefix)
            # blip_str = Translation_Baidu.excute_translation(query=blip_str,from_lang="en", to_lang="zh")
            result.append(blip_str)
        return result

    # 将图片转换成向量
    def _embedding_image(self, image_path: str):
        img = cv2.imread(image_path)
        # 将图像大小调整为VGG16模型的输入大小
        img = cv2.resize(img, (224, 224))
        # 将图像转换为4D张量（样本数量，行，列，通道数）
        img = np.expand_dims(img, axis=0)
        # 预处理图像以适应VGG16模型的输入要求
        img = preprocess_input(img)
        # 提取图像特征
        features = self.cnn_model.predict(img)
        # 将特征展平为一维向量
        vector = features.flatten()
        vector = np.array(vector).tolist()
        return vector

    # 1、在向量数据库进行搜索
    def query_image_by_vector(self, image_path: str):
        vector = self._embedding_image(image_path=image_path)
        # 查询向量数据库
        query_result = self.db_collection.query(
            query_embeddings=vector,
            n_results=self.search_count_from_embedding_db)
        result = []
        blips: list = None
        if len(query_result["metadatas"][0]) > 0 and len(
                query_result["distances"][0]) > 0:
            for i in range(0, len(query_result["distances"][0])):
                if query_result["distances"][0][
                        i] <= self.cos_distance_threshold:
                    result.append(query_result["metadatas"][0][i])
        if len(result) <= 0:
            blips = self._blip_image(image_url=image_path)
            self.blip_info = blips[0]
        return {
            "original_vector": vector,
            "search_result": result,
            "original_blip": blips[0] if blips is not None else None
        }

    # 2、根据图片摘要搜索互联网
    def search_image_by_google(self, text: str) -> list[dict]:

        def download_images(image_urls: list):
            images = []
            for url in image_urls:
                try:
                    image_name = str(uuid.uuid4())
                    local_path = f"{self.destination}\\{image_name}.jpg"
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    with open(local_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    images.append(local_path)
                except requests.exceptions.RequestException as e:
                    continue
            return images

        base_url = "https://www.googleapis.com/customsearch/v1"
        _params = f"key={GOOGLE_APIKEY}&cx={GOOGLE_SEARCH_ID}"
        content = []
        start = (self.page_index - 1) * self.search_count_from_google
        system_params = f"{_params}&q={text}&lr=lang_zh-CN&sort=review-rating:d:s&searchType=image&start={start}"
        search_result = json.loads(
            requests.get(url=f"{base_url}?{system_params}").text)
        if search_result and "items" in search_result:
            for item in search_result["items"]:
                content.append(item["link"])
        images = download_images(content)
        return images

    # 3、将从互联网搜索出来的图片进行摘要提取并与原始输入摘要/文本进行欧式计算，返回相似度最高的前面N条
    def compare_google_and_orignal_image(self, google_result: list,
                                         original_vector):

        def get_distance(google_result: list, original_vector):
            blip_google_results = []
            for item in google_result:
                try:
                    target_vector = self._embedding_image(item)
                    blip_google_results.append({
                        "image_path": item,
                        "vector": target_vector,
                        "distance": 0
                    })
                except:
                    pass
            result = []
            for i in range(0, len(blip_google_results)):
                try:
                    consine_distance = distance.cosine(
                        np.array(original_vector),
                        np.array(blip_google_results[i]["vector"]))
                    blip_google_results[i]["distance"] = consine_distance
                    if consine_distance <= self.cos_distance_threshold:
                        result.append(blip_google_results[i])
                except:
                    pass
            return result, blip_google_results

        satisfy_list, all_search_list = get_distance(google_result,
                                                     original_vector)
        total_search_times = int(self.max_count_from_google /
                                 self.search_count_from_google)
        cur_time_index = 1

        while len(
                satisfy_list
        ) < self.search_count_from_embedding_db and cur_time_index <= total_search_times:

            self.page_index = self.page_index + 1
            google_images = self.search_image_by_google(self.blip_info)
            images, all_list = get_distance(google_images, original_vector)
            satisfy_list.extend(images)
            all_search_list.extend(all_list)
            cur_time_index = cur_time_index + 1

        if len(satisfy_list) < self.search_count_from_embedding_db:
            res_count = self.search_count_from_embedding_db - len(satisfy_list)
            res_list = sorted(all_search_list,
                              key=lambda x: float(x["distance"]),
                              reverse=False)[:res_count]
            satisfy_list.extend(res_list)
        # 删除多余文件
        Image_Search_Util_Img.delete_files(satisfy_list, all_search_list)
        return satisfy_list

    # 4、将图片信息存入向量数据库，并返回存入成功的图片数据
    def embedding_image_info(self, images: list):
        if images is None or len(images) <= 0:
            raise ValueError("images参数必须包含有效值")
        result_images = []
        for image in images:
            if "image_path" not in image or "vector" not in image:
                continue
            id = str(uuid.uuid4())
            self.db_collection.add(
                ids=id,
                embeddings=image["vector"],
                metadatas={"image_path": image["image_path"]})
            result_images.append(image["image_path"])
        return result_images

    def set_image_init_data(self, dic_path: str):
        files = os.listdir(dic_path)
        image_list = []
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(dic_path, file)
            vector = self._embedding_image(file_path)
            image_list.append({"image_path": file_path, "vector": vector})
        return self.embedding_image_info(image_list)

    @staticmethod
    def delete_files(satisfy_list, all_search_list):
        satisfy_image_paths = set(item['image_path'] for item in satisfy_list)

        for item in all_search_list:
            image_path = item['image_path']
            if image_path not in satisfy_image_paths:
                try:
                    os.remove(image_path)
                except OSError as e:
                    pass
