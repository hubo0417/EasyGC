BASE_CONFIG = {
    # 将需要进行向量化的文件上传到这个地址
    "upload_file_base_path":
    "D:\\EasyGC\\application\\utils\\spider_resource\\upload",
    # LLM大模型文件地址
    "llm_model_path": "D:\\ChatGLM2-6B\\chatglm2-6b-model-int4",
    # chroma_db向量数据库文件存放地址
    "chromadb_path": "D:\\ChatGLM2-6B\\knowledge_center\\chroma_db",
    # 向量化模型文件存放地址
    "embedding_model_path": "D:\\Text2Vec",
    # refiner模型文件地址
    "sdxl_refiner_path": "D:\\ChatGLM2-6B\\stable-diffusion-xl-refiner-1.0",
    # base模型文件地址
    "sdxl_base_path": "D:\\ChatGLM2-6B\\stable-diffusion-xl-base-1.0",
    # sdxl的lora模型文件地址
    "sdxl_lora_path": "D:\\ChatGLM2-6B\\stable-diffusion-xl-base-1.0\\lora",
    # 百度翻译的APPID
    "baidu_appid": "20231025001858786",
    # 百度翻译的APPKEY
    "baidu_app_key": "LazJH8nmTXw5YNND_QO4"
}
# base模型相关的vae文件地址
VAE_PATH = f"{BASE_CONFIG['sdxl_base_path']}\\vae_fix"
# 图片生成之后的保存地址
BASE_FILE_PATH = "D:\\ChatGLM2-6B\\knowledge_center"
# 谷歌搜索引擎的key
GOOGLE_APIKEY = ""
# 谷歌搜索引擎的自有ID
GOOGLE_SEARCH_ID = ""
# Azure OpenAI服务的key
AZURE_OPENAI_KEY = ""
# Azure OpenAI服务的终结点
AZURE_OPENAI_ENDPOINT = ""
