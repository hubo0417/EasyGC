from typing import List
from diffusers import DiffusionPipeline, AutoencoderKL
import torch
from threading import RLock
from configs.base_config import BASE_CONFIG, VAE_PATH


class SD_Base_Model:

    model_id: str = BASE_CONFIG["sdxl_base_path"]
    export: bool = True
    single_lock = RLock()
    is_cuda: bool = False  # 是否只在CUDA上运行推理
    n_steps: int = 50  # 迭代去噪步数
    high_noise_frac: float = 0.8  # 噪声优化层度
    guidance_scale: float = 9  # 数值越高越prompt相符合
    H: int = 1024  # 图片的高
    W: int = 1024  # 图片的宽

    def __init__(self, **kwargs) -> None:
        self.init_params(**kwargs)
        self.base_model = None
        self.vae = None

    def init_params(self, **kwargs):
        if "model_id" in kwargs:
            self.model_id = kwargs["model_id"]
        if "is_cuda" in kwargs:
            self.is_cuda = kwargs["is_cuda"]
        if "n_steps" in kwargs:
            self.n_steps = int(kwargs["n_steps"])
            if self.n_steps > 100 or self.n_steps < 30:
                self.n_steps = 40
        if "guidance_scale" in kwargs:
            self.guidance_scale = float(kwargs["guidance_scale"])
            if self.guidance_scale > 8:
                self.guidance_scale = 7.5
        if "high_noise_frac" in kwargs:
            self.high_noise_frac = float(kwargs["high_noise_frac"])
            if self.high_noise_frac > 0.9 or self.high_noise_frac < 0.5:
                self.high_noise_frac = 0.8
        if "H" in kwargs:
            self.H = int(kwargs["H"])
            if self.H > 1024 or self.H < 128:
                self.H = 512
        if "H" in kwargs:
            self.W = int(kwargs["W"])
            if self.W > 1024 or self.W < 128:
                self.W = 512

    # 使用基础模型生成图片 返回PIL图片
    def get_image_to_image_single_prompt(self,
                                         query: str,
                                         image_count: int = 1,
                                         negative_prompt: str = None):
        # seed = 1337
        # generator = torch.Generator("cuda").manual_seed(seed)
        image = self.base_model(query,
                                num_inference_steps=self.n_steps,
                                guidance_scale=self.guidance_scale,
                                num_images_per_prompt=image_count,
                                negative_prompt=negative_prompt,
                                height=self.H,
                                width=self.W).images
        return image

    # 通过提示词集合获取对应图片集合
    def get_base_images_multiple_prompts(self,
                                         prompts: List[str],
                                         image_count: int = 1,
                                         negative_prompt: str = None):
        if len(prompts) <= 0:
            raise ValueError("未能获取到对应提示词")
        negative_prompts: List[str] = []
        for item in prompts:
            negative_prompts.append(negative_prompt)
        images = self.base_model(prompt=prompts,
                                 num_inference_steps=self.n_steps,
                                 guidance_scale=self.guidance_scale,
                                 num_images_per_prompt=image_count,
                                 negative_prompt=negative_prompts,
                                 height=self.H,
                                 width=self.W).images
        return images

    # 在潜在空间生成图片
    def get_base_latent_image_single_prompt(self,
                                            query: str,
                                            image_count: int = 1,
                                            negative_prompt: str = None):
        images = self.base_model(query,
                                 num_inference_steps=self.n_steps,
                                 denoising_end=self.high_noise_frac,
                                 num_images_per_prompt=image_count,
                                 negative_prompt=negative_prompt,
                                 output_type="latent").images
        return images

    # 在潜在空间生成图片
    def get_base_latent_image_multiple_prompts(self,
                                               prompts: List[str],
                                               image_count: int = 1,
                                               negative_prompt: str = None):
        if len(prompts) <= 0:
            raise ValueError("未能获取到对应提示词")
        negative_prompts: List[str] = []
        for item in prompts:
            negative_prompts.append(negative_prompt)
        images = self.base_model(prompts,
                                 num_inference_steps=self.n_steps,
                                 denoising_end=self.high_noise_frac,
                                 num_images_per_prompt=image_count,
                                 negative_prompt=negative_prompts,
                                 output_type="latent").images
        return images

    def unload_model(self):
        if self.base_model is not None:
            self.base_model = None
        torch.cuda.empty_cache()

    def load_model(self):
        self.unload_model()
        if self.base_model is None:
            self.vae = AutoencoderKL.from_pretrained(VAE_PATH,
                                                     torch_dtype=torch.float16)
            # 初始化模型
            self.base_model = DiffusionPipeline.from_pretrained(
                self.model_id,
                vae=self.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16")
            # 在GPU上进行推理
            if self.is_cuda is True:
                self.base_model.to("cuda")
            # GPU+CPU联合推理
            else:
                self.base_model.enable_model_cpu_offload()

    def fuse_lora(self, loras: list):
        if self.base_model is not None:
            self.base_model.unfuse_lora()
            if len(loras) > 0:
                names = []
                scales = []
                for lora in loras:
                    self.base_model.load_lora_weights(
                        BASE_CONFIG["sdxl_lora_path"],
                        weight_name=f"{lora['name']}.safetensors",
                        adapter_name=lora['name'])
                    names.append(lora['name'])
                    scales.append(lora['scale'])
                    self.base_model.fuse_lora(lora_scale=lora["scale"])
                # self.base_model.set_adapters(names, scales)

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(SD_Base_Model, "_instance"):
            with SD_Base_Model.single_lock:
                if not hasattr(SD_Base_Model, "_instance"):
                    SD_Base_Model._instance = cls(*args, **kwargs)
        else:
            SD_Base_Model._instance.init_params(**kwargs)
        return SD_Base_Model._instance
