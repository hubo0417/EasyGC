# 重写继承模型
import os
import shutil
import gradio as gr
from gradio.components.chatbot import Chatbot
from utils.pipeline import Pipeline_Item, Pipeline_Process_Task
from utils.generate.note_generate_utils import Note_Generate_Utils
from llms.llm_helper import ChatGLM_Helper
from agents.agent_controller.api_sequence_agent import API_Sequence_Agent
from agents.agent_executor.sequence_agentexecutor import Sequence_AgentExecutor
from sdxl.generate_style_config import style_list
from embeddings.embedding_helper import EmbeddingHelper
from utils.tool_sets import Tool_Sets
from utils.lora_sets import Lora_Sets
from configs.base_config import BASE_CONFIG, BASE_FILE_PATH

uploaded_image_url = ""
llm_helper = ChatGLM_Helper.instance()


def init_agent(tool_names: list):
    if len(tool_names) > 0:
        tools = []
        for name in tool_names:
            tools.append(Tool_Sets.init_tools(name, llm=llm_helper.llm))
        agent = API_Sequence_Agent(tools=tools, llm=llm_helper.llm)
        agent_exec = Sequence_AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, max_iterations=1)
        return agent_exec
    return None


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = '<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)

    return text


# def join_history(is_note: bool = False):
#     result = ""
#     for item in history:
#         for key, value in item.items():
#             result += f"{value}\n"
#     return result

style_name = [(i["name"]) for i in style_list]
css = """
    #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        },
    #upload_tool{
        height:96px !important
        width:280px !important
    }
    #content_type{
        flex-grow:4 !important
    }

"""
with gr.Blocks(css=css) as web_gc:
    gr.HTML("""<h1 align="center">EasyGC</h1>""")
    pipe_state = gr.State()
    with gr.Tab("text generate"):
        with gr.Row(variant="panel"):
            with gr.Column(scale=10):
                with gr.Row():
                    chatbot = gr.Chatbot()
                with gr.Row():
                    user_input = gr.Textbox(show_label=False,
                                            placeholder="Input...",
                                            lines=4).style(container=False)
            with gr.Column(scale=2):
                files = gr.Files(label="上传文件", height=110)
                author_input = gr.Textbox(lines=1,
                                          show_label=True,
                                          placeholder="文档元数据标记，用于后续关键字搜索",
                                          label="文档源",
                                          interactive=True)
                number = gr.Number(value=1000, label="切分长度", minimum=100)
                overlap = gr.Number(value=0, label="重叠长度", minimum=0)
                lode_Btn = gr.Button("处理文件", variant="primary")
                submitBtn = gr.Button("发送信息", variant="primary", min_width=60)
        with gr.Row():
            with gr.Column(scale=10):
                content_type = gr.Dropdown(
                    choices=[tool["name"] for tool in Tool_Sets.load_tools()],
                    type="value",
                    multiselect=True,
                    label="工具",
                    interactive=True)
            with gr.Column(scale=10):
                upload_tool = gr.File(label="上传工具",
                                      height=94,
                                      elem_id="upload_tool")
            with gr.Column(scale=2):
                handle_tools = gr.Button("加载工具",
                                         variant="primary",
                                         min_width=60,
                                         width=100)
        with gr.Row():
            # note_Btn = gr.Button("生成文章", variant="primary")
            emptyBtn = gr.Button("清除历史会话")
            load_llm_model = gr.Button("重新加载LL模型", variant="primary")
    with gr.Tab("txt2img"):
        with gr.Row():
            with gr.Column(scale=6):
                gallery = gr.Gallery(label="图片生成",
                                     show_label=False,
                                     elem_id="gallery",
                                     grid=[4])
            with gr.Column(scale=3):
                lora_upload = gr.Textbox(
                    lines=1,
                    show_label=True,
                    placeholder="填写模型完整的绝对路径如： D:/aaa/xxxx.safetensors",
                    label="模型路径",
                    interactive=True)
                scale = gr.Number(value=1, label="权重", minimum=0.1, maximum=1)
                order_num = gr.Number(value=0, label="排序", minimum=0)
                tag_words = gr.Textbox(lines=1,
                                       show_label=True,
                                       placeholder="模型触发词，用逗号分隔",
                                       label="触发词",
                                       interactive=True)
                handle_lora = gr.Button("加载模型",
                                        variant="primary",
                                        min_width=60,
                                        width=100)
        with gr.Row():
            style_dropdown = gr.Dropdown(choices=style_name,
                                         type="value",
                                         value="",
                                         show_label=False,
                                         container=False,
                                         max_choices=1,
                                         multiselect=False,
                                         interactive=True)
            lora_dropdown = gr.Dropdown(choices=Lora_Sets.load_loras(),
                                        type="value",
                                        show_label=False,
                                        container=False,
                                        max_choices=2,
                                        multiselect=True,
                                        interactive=True)

        with gr.Row():
            comment = gr.Textbox(lines=2,
                                 show_label=True,
                                 placeholder="",
                                 label="图片描述",
                                 interactive=True)
            ok_note_Btn = gr.Button("生成", variant="primary")
    with gr.Tab("img2img"):
        with gr.Row():
            with gr.Column(scale=6):
                img_input = gr.Image(source="upload",
                                     show_label=False,
                                     interactive=True,
                                     type="filepath")

            with gr.Column(scale=6):
                img_output = gr.Gallery(label="Generated images",
                                        show_label=False,
                                        elem_id="img2img_gallery",
                                        grid=[4])
        with gr.Row():
            style_dropdown_img2img = gr.Dropdown(choices=style_name,
                                                 type="value",
                                                 value="",
                                                 show_label=False,
                                                 container=False,
                                                 max_choices=1,
                                                 multiselect=False,
                                                 interactive=True)
            img_modify_comment = gr.Textbox(show_label=False,
                                            placeholder="关键词/句...",
                                            lines=5).style(container=False)
        with gr.Row():
            img2img_btn = gr.Button("生成图片", variant="primary")

    def handle_upload_lora(scale, order_num, tag_words, lora_upload):
        if lora_upload and tag_words and os.path.exists(lora_upload):
            loras = Lora_Sets.init_lora(order_num=order_num,
                                        scale=scale,
                                        tag_words=tag_words,
                                        model_path=lora_upload)
            new_drop_down = lora_dropdown.update(choices=loras)
            return new_drop_down

    def handle_upload_tools(upload_tool):
        file_path = upload_tool.name
        tools = Tool_Sets.regist_tool(sorted=1, file_path=file_path)
        tool_result = [tool["name"] for tool in tools]
        new_drop_down = content_type.update(choices=tool_result)
        return new_drop_down

    def handle_files(files, author_input, number, overlap):
        upload_file_base_path = BASE_CONFIG["upload_file_base_path"]
        files_path_array: list = []
        for file_obj in files:
            # 将文件复制到临时目录中
            shutil.copy(file_obj.name, upload_file_base_path)
            # 获取上传Gradio的文件名称
            file_name = os.path.basename(file_obj.name)
            # 获取拷贝在临时目录的新的文件地址
            files_path_array.append(
                os.path.join(upload_file_base_path, file_name))
        for file_path in files_path_array:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            texts = EmbeddingHelper.splitText(
                content,
                chunk_size=number if number else 1000,
                overlap=overlap if overlap else 0)
            item = {}
            item["content"] = content
            item["source"] = author_input if author_input else "未知"
            helper = EmbeddingHelper(collection_name="literature")
            helper.embedding_texts(texts=texts, metadatas=[item])

    def _load_llm_model():
        if llm_helper.llm.model is None:
            llm_helper.llm.load_model(model_name_or_path=llm_helper.model_id)

    def generate_image_by_image(img_input, img_modify_comment,
                                style_dropdown_img2img):
        style = {}
        for i in style_list:
            if i["name"] == style_dropdown_img2img:
                style = i
                break
        util = Note_Generate_Utils(llm=llm_helper.llm,
                                   base_file_path=BASE_FILE_PATH)
        # 注册图片生成方法
        item = Pipeline_Item(Obj=util,
                             Method="generate_images_by_image",
                             Is_Use_Pre_Result=False,
                             Params={
                                 "image_url": img_input,
                                 "style": style,
                                 "text": img_modify_comment
                             })
        pipe = Pipeline_Process_Task()
        pipe.add_item(item=item)
        images = pipe.execute_pipeline()
        return images

    def generate_image(comment, style_dropdown, lora_dropdown):
        note = comment
        if pipe_state.value:
            images = pipe_state.value.continue_pipeline(note)
        else:
            style = {}
            for i in style_list:
                if i["name"] == style_dropdown:
                    style = i
                    break
            loras = []
            for lora_item in sorted(
                    Lora_Sets.load_loras(is_only_return_name=False),
                    key=lambda x: int(x["sored"]),
                    reverse=True):
                if lora_item["name"] in lora_dropdown:
                    loras.append(lora_item)
            util = Note_Generate_Utils(llm=llm_helper.llm,
                                       base_file_path=BASE_FILE_PATH)
            pipe = Pipeline_Process_Task()
            # 注册图片介绍文字提取方法
            item = Pipeline_Item(Obj=util,
                                 Method="get_image_flags",
                                 Is_Use_Pre_Result=True)
            pipe.add_item(item=item)

            # 注册图片生成方法
            item = Pipeline_Item(Obj=util,
                                 Method="generate_images",
                                 Is_Use_Pre_Result=True)
            pipe.add_item(item=item)
            images = pipe.execute_pipeline({
                "content": note,
                "style": style,
                "loras": loras
            })

        return images

    def _excute_agent_predict(user_input, chatbot: Chatbot, content_type):
        prompt = user_input
        try:
            agent_exec = init_agent(content_type)
            if agent_exec is not None:
                agent_generate = agent_exec.run(prompt)
                # 没有找到合适的工具满足用户的输入信息
                if agent_generate == "无工具" or agent_generate[
                        "sumary_result"] is None:
                    yield None
                else:
                    chatbot.append((parse_text(prompt), ""))
                    content_generate = agent_generate["sumary_result"][
                        "content"]
                    if "original" in agent_generate["sumary_result"]:
                        chatbot[-1] = (parse_text(
                            prompt
                        ), f"原文：{agent_generate['sumary_result']['original']}\n"
                                       )
                    if isinstance(content_generate, list):
                        content = ""
                        for item in content_generate:
                            content = f"<a href=\"{item['link']}\">{item['title']}</a>\n{item['snippet']}\n\n------------------------\n\n"
                            chatbot[-1] = (parse_text(prompt),
                                           f"{chatbot[-1][1]}{content}")
                            yield chatbot
                    else:
                        for token in content_generate:
                            # 采用mapreducedocumentchain时返回的迭代器中包含output_text
                            if "output_text" in token:
                                for doc_token in token["output_text"]:
                                    chatbot[-1] = (
                                        parse_text(prompt),
                                        f"{chatbot[-1][1]}{doc_token}")
                                    yield chatbot
                            # 采用StreamChain时，直接返回generate token
                            else:
                                chatbot[-1] = (parse_text(prompt),
                                               f"{chatbot[-1][1]}{token}")
                                yield chatbot
                    if "pipe" in agent_generate:
                        pipe_state.value = agent_generate["pipe"]
        except Exception:
            pass

    def _excute_llm_predict(user_input, chatbot: Chatbot, content_type):
        prompt = user_input
        chatbot.append((parse_text(prompt), ""))
        for token in llm_helper.llm._stream(prompt,
                                            temperature=0.95,
                                            top_p=0.7):
            chatbot[-1] = (parse_text(user_input),
                           f"{chatbot[-1][1]}{token.text}")
            yield chatbot

    def predict(user_input, chatbot: Chatbot, content_type):
        is_need_llm_predict: bool = False
        if user_input:
            if content_type:
                for token in _excute_agent_predict(user_input, chatbot,
                                                   content_type):
                    if token is None:
                        is_need_llm_predict = True
                        break
                    yield token
                if is_need_llm_predict is True:
                    for token in _excute_llm_predict(user_input, chatbot,
                                                     content_type):
                        yield token
            else:
                for token in _excute_llm_predict(user_input, chatbot,
                                                 content_type):
                    yield token

    def reset_user_input():
        return gr.update(value='')

    def reset_state():
        return [], [], [], []

    load_llm_model.click(_load_llm_model, show_progress=True)

    ok_note_Btn.click(generate_image,
                      inputs=[comment, style_dropdown, lora_dropdown],
                      outputs=[gallery],
                      show_progress=True,
                      queue=True)
    submitBtn.click(predict,
                    inputs=[user_input, chatbot, content_type],
                    outputs=[chatbot],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state,
                   outputs=[chatbot, content_type, files, upload_tool],
                   show_progress=True)
    img2img_btn.click(
        generate_image_by_image,
        inputs=[img_input, img_modify_comment, style_dropdown_img2img],
        outputs=[img_output])
    lode_Btn.click(fn=handle_files,
                   inputs=[files, author_input, number, overlap],
                   show_progress=True)
    handle_tools.click(fn=handle_upload_tools,
                       inputs=[upload_tool],
                       outputs=[content_type],
                       show_progress=True)
    handle_lora.click(fn=handle_upload_lora,
                      inputs=[scale, order_num, tag_words, lora_upload],
                      outputs=[lora_dropdown],
                      show_progress=True)
web_gc.queue().launch(share=True, inbrowser=True)
