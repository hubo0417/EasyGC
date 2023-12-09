from embeddings.embedding_helper import EmbeddingHelper
from utils.generate.note_generate_utils import Note_Generate_Utils


class Note_Generate_Utils_Literature(Note_Generate_Utils):
    prompt_template = """你现在是一名专业的短视频编剧，请根据已知信息：'{text}'
                        创作一篇分镜脚本，脚本中要包含'画面'和'旁白'两个元素，
                        至少输出8个画面以及8段旁白，并且必须按照以下格式进行输出：

                        画面1：【xxxxxxxxxxxx】
                        旁白1：画面1对应的旁白信息

                        画面2：【xxxxxxxxxxxx】
                        旁白2：画面2对应的旁白信息

                        ...

                        """

    def __init__(self,
                 llm,
                 base_file_path: str = None,
                 max_image_count: int = 9) -> None:
        super().__init__(llm=llm,
                         base_file_path=base_file_path,
                         max_image_count=max_image_count)

    def read_note_list(self, **kwargs):
        try:
            article = kwargs["article"]
            top = int(kwargs["top"])
            helper = EmbeddingHelper(collection_name="literature")
            datas = helper.query(article,
                                 top,
                                 is_find_metedata=True
                                 # where_document={"$contains": article}
                                 )
            return datas
        except Exception:
            return None

    def summary_response(self, details: list = None) -> str:
        text: str = ""
        if details is not None and len(details) > 0:
            for i in range(0, len(details)):
                cur_text = details[i]['content']
                text += f"\n{cur_text}\n"
            return self.generate_article_content(
                text=text,
                prompt_template=self.prompt_template,
                is_only_return_result=False)
        else:
            return None
