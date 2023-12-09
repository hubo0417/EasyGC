import os
import json
import shutil
from configs.base_config import BASE_CONFIG


class Lora_Sets:

    @classmethod
    def load_loras(cls, is_only_return_name: bool = True):
        current_dir = os.path.dirname(__file__)
        # 向上一级移动两级，得到项目根目录
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        # 构建相对路径
        relative_path = os.path.join(project_root, "configs", 'loras.json')
        config_data = []
        with open(relative_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
        if is_only_return_name is True:
            config_data = [item["name"] for item in config_data]
        return config_data

    @classmethod
    def init_lora(cls, order_num: int, scale: float, tag_words: str,
                  model_path: str):
        file_name = os.path.basename(model_path)
        file_name, _ex = os.path.splitext(file_name)
        lora = {
            "name": file_name,
            "tag_words": tag_words,
            "scale": scale,
            "sored": order_num
        }
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))
        relative_path = os.path.join(project_root, "configs", 'loras.json')
        with open(relative_path, 'r+', encoding='utf-8') as file:
            config_data = json.load(file)
            # 检查模型名称是否重复
            name_is_exist = False
            for data in config_data:
                if file_name == data["name"]:
                    name_is_exist = True
                    break
            if name_is_exist is False:
                config_data.append(lora)
                file.seek(0)
                json.dump(config_data, file, indent=4)
        if name_is_exist is True:
            raise ValueError("模型名称已经存在，请修改后重新上传")
        lora_path = BASE_CONFIG["sdxl_lora_path"]
        shutil.copy2(model_path, lora_path)
        return cls.load_loras()
