# 一、说明
详细描述请参见 知乎：chatglm2-2b+sdxl1.0+langchain打造私有AIGC（六）-完结[https://zhuanlan.zhihu.com/p/669430175]

1.开源版本的LLM，是基于ChatGLM2-6B-INT4（运行时显存暂用：约6G）的量化模型进行开发的，因为考虑到大多数人的显卡并没有支持全量模型的能力。如果你的显卡够好，请自己替换成全量模型，甚至32K模型（我开发的时候使用的32K模型ChatGLM2-2B-32K，运行时显存占用：13G）

2.如果不使用ChatGLM2-6B-INT4的量化模型，可能会导致应用内的提示词与模型不能完美契合的情况，尤其是在Agent模块，让LLM判断使用哪些工具的场景。遇到这种情况，请自行修改提示词

3.开源仓库中只包含应用源码，不包含LLM，Embedding，SDXL的模型文件，如有需要模型文件的朋友可以留言，我私发，也可以自行去huggingface下载

# 二、功能介绍
应用分为3个模块

## 1、文本生成模块
在这个模块里主要有三个功能，一是对话功能，二是上传文本文件进行向量化的功能，三是上传自定义工具功能
![image](https://github.com/hubo0417/EasyGC/assets/17717096/3f122b51-b0d1-4737-8263-8a050242bd64)
1.1、对话功能

在文本框输入信息，点击【发送信息】按钮实现与AI对话

1.2、上传文本文件功能

在页面最右边，拖入事先处理好的文本文件（建议是txt文件），设置好参数，点击【处理文件】。便可将文档中的内容进行向量化处理（向量化后的文档，目前需配合工具使用，可自己修改源码直接使用）

1.3、上传自定义工具

在页面输入框的下方，有一个【工具】的下拉框，还有一个【上传工具】的上传组件，将工具拖入到上传组件中，点击【加载工具】，便可将工具植入到应用中，在下来框中出现对应选项。（工具其实就是.py文件，自己可以参照源码，写一个自己的.py文件，上传之后就可以应用便会加载你的.py文件）



模块思路：在这个页面，如果【工具】下拉框选择了一个或多个工具，当点击【发送消息】后，应用会先让LLM判断是否有合适的工具来处理用户输入，如果判断出没有合适的工具则会让LLM直接回答用户的输入，如果有合适的工具，则会调用对应的工具

工具思路：1/根据用户输入信息在工具中提取关键词，2/根据关键词到向量数据库查找对应文档，3/根据对应文档内容让LLM生成目标内容

如果对以上内容不能理解，请翻阅我之前的文章，里面有介绍整个应用的流程

【清除历史会话】：应用中的每次非工具调用情况下的问答，都是将整个历史对话记录当成输入上下文，传给LLM模型，因为LLM模型的token限制，有时候会导致索引报错，所以有此功能

【重新加载LLM模型】：因为显存有限，一台电脑中最好只加载一个模型，当使用SDXL模型生成图片的时候，应用会自动卸载掉之前加载的LLM模型（LLM模型在应用启动时自动加载），所以设置了重新加载LLM模型功能

## 2、文生图模块

![image](https://github.com/hubo0417/EasyGC/assets/17717096/50ddd168-689a-4f58-ae12-606b5cfad307)

在【图片描述】输入图片描述，务必用【】将描述内容括起来（一条描述内容用一个【】），如果你不喜欢使用【】请自行在源码中修改

在图片描述的上方，有个风格选择下拉框（chatglm2-2b+sdxl1.0+langchain打造私有AIGC（五））。

旁边还有一个多选的下拉框，用于选择加载哪些loRA模型。对于如何将lora模型嵌入到应用中，这个功能我会在下个版本中添加，目前可以手动加模型放置到本地电脑的某个磁盘文件夹下，再到应用中修改两处配置文件即可，下面讲解代码结构的时候会讲到

点击【生成】按钮后，应用默认会为每一条描述内容，生成2张图片

## 3、图生图模块

![image](https://github.com/hubo0417/EasyGC/assets/17717096/f6042a78-098e-4e2e-8fc5-98b69fe1da56)

1、界面左侧上传原始图片，2、在界面左下方选择图片风格，3、界面右下角输入关键字，4、点击生成图片

应用默认是为每一张原始图生成4张最终图，特别强调一下，此模块主要是在修改图片的风格，对于在图片内容的修改目前并不能实现

在这个模块里面，描述内容可以不用加【】

图生图的具体效果：


![image](https://github.com/hubo0417/EasyGC/assets/17717096/3824b153-1983-4960-9be9-6fb301da380a)


![image](https://github.com/hubo0417/EasyGC/assets/17717096/dee900d4-2d9d-44b9-a328-d9bb438a1208)


![image](https://github.com/hubo0417/EasyGC/assets/17717096/3b848e33-2e06-4f94-8052-822d37e6d079)

# 三、代码结构介绍
![新建 XLS 工作表_Sheet1](https://github.com/hubo0417/EasyGC/assets/17717096/2ecfe24c-8240-4353-af1b-197be0deaf45)

# 四、部署方式
## 1、基础环境搭建参照下面连接进行CUDA，pytorch等基础环境设置

[chatglm2-2b+sdxl1.0+langchain打造私有AIGC（一）]
(https://zhuanlan.zhihu.com/p/665933712)

## 2、通过pip安装requirements.txt文件里面所罗列的依赖包

## 3、修改配置文件（configs/base_config.py）

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
    "baidu_appid": "",
    # 百度翻译的APPKEY
    "baidu_app_key": ""
}
"#base模型相关的vae文件地址"
VAE_PATH = f"{BASE_CONFIG['sdxl_base_path']}\\vae_fix"
"#3图片生成之后的保存地址"
BASE_FILE_PATH = "D:\\ChatGLM2-6B\\knowledge_center"


请根据自己实际情况按注释说明修改配置值

## 4、在cmd中直接执行 python web.py 启动应用
