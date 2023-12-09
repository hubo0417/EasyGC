from typing import List
from sdxl.sd_base_model import SD_Base_Model
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import torch
from threading import RLock
from configs.base_config import BASE_CONFIG


class SD_Refiner_Model:

    model_id: str = BASE_CONFIG["sdxl_refiner_path"]
    export: bool = True
    single_lock = RLock()
    is_cuda: bool = False
    n_steps: int = 50
    high_noise_frac: float = 0.8
    guidance_scale: float = 9  # 数值越低，画面效果越趋近于抽象油画
    is_combine_base: bool = True
    H: int = 1024  # 图片的高
    W: int = 1024  # 图片的宽

    def __init__(self, **kwargs) -> None:
        self.init_params(**kwargs)
        self.base_model: SD_Base_Model = None
        self.refiner_model = None

    def init_params(self, **kwargs):
        if "model_id" in kwargs:
            self.model_id = kwargs["model_id"]
        if "is_cuda" in kwargs:
            self.is_cuda = bool(kwargs["is_cuda"])
        if "n_steps" in kwargs:
            self.n_steps = int(kwargs["n_steps"])
            if self.n_steps > 100 or self.n_steps < 30:
                self.n_steps = 40
        if "guidance_scale" in kwargs:
            self.guidance_scale = float(kwargs["guidance_scale"])
            if self.guidance_scale > 8:
                self.guidance_scale = 7.5
        if "is_combine_base" in kwargs:
            self.is_combine_base = bool(kwargs["is_combine_base"])
        if "H" in kwargs:
            self.H = int(kwargs["H"])
            if self.H > 1024 or self.H < 128:
                self.H = 1024
        if "H" in kwargs:
            self.W = int(kwargs["W"])
            if self.W > 1024 or self.W < 128:
                self.W = 1024

    # 通过基础图片+单条提示词得到另一些精炼加工图片
    def get_image_to_image_single_prompt(self,
                                         query: str,
                                         image_url: str = None,
                                         image_count: int = 1,
                                         negative_prompt: str = None):
        # 获取潜在空间的图片通过单条提示词
        def _get_base_latent_images_single_prompt(query: str,
                                                  image_count: int = 1,
                                                  negative_prompt: str = None):
            if self.base_model is not None:
                images = self.base_model.get_base_latent_image_single_prompt(
                    query=query,
                    image_count=image_count,
                    negative_prompt=negative_prompt)
                return images
            else:
                return None

        target_size: tuple[int, int] = (self.H, self.W)
        if image_url is None and self.is_combine_base is True:
            init_images = _get_base_latent_images_single_prompt(
                query, image_count, negative_prompt)
            images = self.refiner_model(prompt=query,
                                        num_inference_steps=self.n_steps,
                                        denoising_start=self.high_noise_frac,
                                        num_images_per_prompt=image_count,
                                        negative_prompt=negative_prompt,
                                        image=init_images,
                                        target_size=target_size).images
        else:

            init_image = load_image(image_url).convert("RGB")
            images = self.refiner_model(prompt=query,
                                        image=init_image,
                                        num_inference_steps=self.n_steps,
                                        guidance_scale=self.guidance_scale,
                                        negative_prompt=negative_prompt,
                                        num_images_per_prompt=image_count,
                                        target_size=target_size).images
        return images

    # 通过基础图片+多条提示词得到另一些精炼加工图片
    def get_image_to_image_multiple_prompts(self,
                                            prompts: List[str],
                                            image_count: int = 1,
                                            negative_prompt: str = None):

        target_size: tuple[int, int] = (self.H, self.W)

        # 获取潜在空间的图片通过多条提示词
        def _get_base_latent_image_multiple_prompts(
                prompts: List[str],
                image_count: int = 1,
                negative_prompt: str = None):
            if self.base_model is not None:
                images = self.base_model.get_base_latent_image_multiple_prompts(
                    prompts=prompts,
                    image_count=image_count,
                    negative_prompt=negative_prompt)
                return images
            else:
                return None

        if self.is_combine_base is True:
            negative_prompts: List[str] = []
            for item in prompts:
                negative_prompts.append(negative_prompt)
            init_images = _get_base_latent_image_multiple_prompts(
                prompts=prompts,
                image_count=image_count,
                negative_prompt=negative_prompt)

            images = self.refiner_model(
                prompt=prompts,
                num_inference_steps=self.n_steps,
                denoising_start=self.high_noise_frac,
                num_images_per_prompt=image_count,
                image=init_images,
                target_size=target_size,
                negative_prompt=negative_prompts).images
        else:
            raise ValueError("REFINER模型并未定义成需要和BASE模型一起使用")
        return images

    def unload_model(self):
        if self.refiner_model is not None:
            self.refiner_model = None
        if self.base_model is not None:
            if self.base_model.base_model is not None:
                self.base_model.base_model = None
        torch.cuda.empty_cache()

    def load_model(self):
        self.unload_model()
        if self.is_combine_base is True:
            # 处理baseModel
            if self.base_model is None:
                self.base_model = SD_Base_Model.instance(
                    n_steps=self.n_steps,
                    high_noise_frac=self.high_noise_frac,
                    is_cuda=self.is_cuda,
                    H=self.H / 2,
                    W=self.W / 2)
            if self.base_model.base_model is None:
                self.base_model.load_model()
            # 处理refinerModel
            if self.refiner_model is None:
                self.refiner_model = DiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                    text_encoder_2=self.base_model.base_model.text_encoder_2,
                    vae=self.base_model.base_model.vae)
        else:
            self.base_model = None
            if self.refiner_model is None:
                self.refiner_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True)
        if self.is_cuda is True:
            self.refiner_model.to("cuda")
        else:
            self.refiner_model.enable_model_cpu_offload()

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(SD_Refiner_Model, "_instance"):
            with SD_Refiner_Model.single_lock:
                if not hasattr(SD_Refiner_Model, "_instance"):
                    SD_Refiner_Model._instance = cls(*args, **kwargs)
        else:
            SD_Refiner_Model._instance.init_params(**kwargs)
        return SD_Refiner_Model._instance
