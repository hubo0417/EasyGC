import gradio as gr
import os
import shutil
from PIL import Image
from sdxl.generate_style_config import style_list
from sdxl.sd_refiner_model import SD_Refiner_Model
from utils.image_search.image_search_util import Image_Search_Util

style_name = [(i["name"]) for i in style_list]

with gr.Blocks() as web_gc:
    selected_image: str = None
    gr.HTML("""<h1 align="center">EasyGC_Image_Search_Generation</h1>""")
    with gr.Row():
        # 左侧搜索部分
        with gr.Column(scale=6):
            with gr.Row():
                gallery_search = gr.Gallery(label="图像搜索结果",
                                            show_label=False,
                                            elem_id="gallery_search")
            with gr.Row():
                with gr.Column():
                    comment = gr.Textbox(lines=3,
                                         show_label=False,
                                         interactive=True)
                with gr.Column():
                    img = gr.Image(show_label=False,
                                   height=120,
                                   interactive=True,
                                   type="filepath")
            search_note_Btn = gr.Button("搜索", variant="primary")
        # 右侧生成部分
        with gr.Column(scale=6):
            gallery_generate = gr.Gallery(label="图像生成结果",
                                          show_label=False,
                                          elem_id="gallery_generate")
            img_refiner = gr.Image(show_label=False,
                                   height=120,
                                   interactive=True,
                                   type="filepath")
            style_dropdown = gr.Dropdown(choices=style_name,
                                         type="value",
                                         value="",
                                         show_label=True,
                                         container=False,
                                         multiselect=False,
                                         interactive=True)

            ok_note_Btn = gr.Button("生成", variant="primary")

    def search_images(img, comment):

        text: str = None
        if img:
            image_util = Image_Search_Util()
            blips = image_util.blip_image(image_url=img)
            text = blips[0]
        elif comment:
            image_util = Image_Search_Util(search_way="text")
            text = comment
        else:
            raise ValueError("图片与文本至少需要保证有一个值不为空")

        embedding_images = image_util.search_embedding_by_text(text)
        if embedding_images is None or len(embedding_images) <= 0:
            google_images = image_util.search_image_by_google(text)
            compare_result = image_util.compare_google_and_orignal_blipinfo(
                google_result=google_images, original_text=text)
            embedding_images = image_util.embedding_image_info(
                images=compare_result)
        return embedding_images

    def generate_image_by_finner(img_refiner, style_dropdown):
        Image_Search_Util.resize_image(image_path=img_refiner)
        if img_refiner is not None:
            style = {}
            for i in style_list:
                if i["name"] == style_dropdown:
                    style = i
                    break
            sd_model = SD_Refiner_Model().instance(is_combine_base=False)
            sd_model.load_model()
            prompt = style["prompt"].format(prompt="").lower()
            negative_prompt = style["negative_prompt"]
            images = sd_model.get_image_to_image_single_prompt(
                query=prompt,
                image_url=img_refiner,
                image_count=4,
                negative_prompt=negative_prompt)
            # 关闭SDXL模型，释放显卡资源
            sd_model.unload_model()
            return images
        return None

    search_note_Btn.click(search_images,
                          inputs=[img, comment],
                          outputs=[gallery_search],
                          show_progress=True)
    ok_note_Btn.click(generate_image_by_finner,
                      inputs=[img_refiner, style_dropdown],
                      outputs=[gallery_generate])
web_gc.queue().launch(share=True, inbrowser=True)
