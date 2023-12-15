from threading import RLock
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from configs.base_config import BASE_CONFIG


class Blip_Model:
    model_id: str = BASE_CONFIG["blip_model_path"]
    single_lock = RLock()
    is_cuda: bool = False  # 是否只在CUDA上运行推理
    processor = None
    model = None

    def __init__(self) -> None:
        self.load_model()

    def load_model(self):
        processor = BlipProcessor.from_pretrained(self.model_id)
        if self.is_cuda is True:
            model = BlipForConditionalGeneration.from_pretrained(
                self.model_id).to("cuda")
        else:
            model = BlipForConditionalGeneration.from_pretrained(self.model_id)
        self.processor = processor
        self.model = model

    def generate_text_from_image(self,
                                 image_url: str = None,
                                 text: str = None):
        if image_url is None:
            raise ValueError("图片路径为空")
        raw_image = Image.open(image_url).convert("RGB")
        if text is None:
            inputs = self.processor(raw_image, return_tensors="pt")
        else:
            inputs = self.processor(raw_image, text, return_tensors="pt")
        out = self.model.generate(**inputs)
        result = self.processor.decode(out[0], skip_special_tokens=True)
        return result

    def unload_model(self):
        if self.model is not None:
            self.model = None
        if self.processor is not None:
            self.processor = None
        torch.cuda.empty_cache()

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(Blip_Model, "_instance"):
            with Blip_Model.single_lock:
                if not hasattr(Blip_Model, "_instance"):
                    Blip_Model._instance = cls(*args, **kwargs)
        return Blip_Model._instance
