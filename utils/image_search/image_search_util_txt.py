import json
import os
import uuid
from typing import Union
import requests
from blip.blip_model import Blip_Model
from embeddings.embedding_helper import EmbeddingHelper
from configs.base_config import GOOGLE_APIKEY, GOOGLE_SEARCH_ID
from scipy.spatial import distance
from PIL import Image


class Image_Search_Util_Txt:
    search_count_from_google: int = 10
    search_count_from_embedding_db: int = 2
    collection_name = "images_blip_source"
    destination: str = "D:\\EasyGC\\images_download\\blips"
    return_top_n = 0.25

    def __init__(self, **kwargs) -> None:
        if "search_count_from_google" in kwargs:
            self.search_count_from_google = int(
                kwargs["search_count_from_google"])
        if "search_count_from_embedding_db" in kwargs:
            self.search_count_from_embedding_db = int(
                kwargs["search_count_from_embedding_db"])
        if "return_top_n" in kwargs:
            self.return_top_n = float(kwargs["return_top_n"])

    # 1、用图片搜索的时候，提取图片文本摘要
    def blip_image(self, image_url: Union[list, str], prefix: str = None):
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
            result.append(blip_str)
        return result

    # 1/2、文本/图片摘要搜索向量数据库
    def search_embedding_by_text(self, text: str):
        final_data: list = None
        embeddings = EmbeddingHelper(collection_name=self.collection_name)
        where_document = {"$contains": text}
        search_result = embeddings.query(
            message=text,
            count=self.search_count_from_embedding_db,
            is_find_metedata=True,
            filter={"source": "images"},
            where_document=where_document)
        if search_result and len(search_result) > 0:
            final_data = [item["image_path"] for item in search_result]
        return final_data

    # 3、根据文本/图片摘要搜索互联网
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
        system_params = f"{_params}&q={text}&lr=lang_zh-CN&sort=review-rating:d:s&searchType=image"
        page_count = int(self.search_count_from_google / 10)
        for i in range(0, page_count):
            start = i * 10 + 1
            system_params = f"{system_params}&start={start}"
            search_result = json.loads(
                requests.get(url=f"{base_url}?{system_params}").text)
            if search_result and "items" in search_result:
                for item in search_result["items"]:
                    content.append(item["link"])
        images = download_images(content)
        return images

    # 4、将从互联网搜索出来的图片进行摘要提取并与原始输入摘要/文本进行余弦距离计算，返回相似度最高的前面N条
    def compare_google_and_orignal_blipinfo(self, google_result: list,
                                            original_text: str):
        blip_google_results = []
        blip_model = Blip_Model()
        for item in google_result:
            blip_info = blip_model.generate_text_from_image(image_url=item)
            blip_google_results.append({
                "blip_info": blip_info,
                "image_path": item,
                "distance": 0
            })
        helper = EmbeddingHelper(collection_name=self.collection_name)
        orignal_tensor = helper.embeddingModel.embed_query(original_text)
        for i in range(0, len(blip_google_results)):
            google_tensor = helper.embeddingModel.embed_query(
                blip_google_results[i]["blip_info"])
            consine_distance = 1 - distance.cosine(google_tensor,
                                                   orignal_tensor)
            blip_google_results[i]["distance"] = consine_distance
        num = int(self.search_count_from_google * self.return_top_n)
        result = sorted(blip_google_results,
                        key=lambda x: x["distance"],
                        reverse=True)[:num]
        return result

    # 5、将图片信息存入向量数据库，并返回存入成功的图片数据
    def embedding_image_info(self, images: list):
        if images is None or len(images) <= 0:
            raise ValueError("images参数必须包含有效值")
        helper = EmbeddingHelper(collection_name=self.collection_name)
        result_images = []
        for image in images:
            if "image_path" not in image or "blip_info" not in image:
                continue
            item = {}
            item["source"] = "images"
            item["image_path"] = image["image_path"]
            helper.embedding_texts(texts=[image["blip_info"]],
                                   metadatas=[item])
            result_images.append(image["image_path"])
        return result_images

    @staticmethod
    def resize_image(image_path: str):
        # 判断是否存在图片
        is_exist = os.path.exists(image_path) and os.path.isfile(image_path)
        if is_exist is False:
            raise ValueError("图片地址错误，请检查图片是否存在")
        # 图片路径
        output_image_path = image_path
        # 打开原始图片
        original_image = Image.open(image_path)
        # 获取原始图片尺寸
        width, height = original_image.size

        # 逐步计算图片大小（按等比例放缩）
        if width * height > 1024 * 1024:
            size_is_approve: bool = False
            resize_step = 0.1
            while size_is_approve is False:
                width = int(width * (1 - resize_step))
                height = int(height * (1 - resize_step))
                size_is_approve = width * height < 1024 * 1024
                resize_step = resize_step + 0.1
            # 调整图片大小
            resized_image = original_image.resize((width, height))
            # 保存调整后的图片
            resized_image.save(output_image_path)
            # 关闭图像
        original_image.close()
