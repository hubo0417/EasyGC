from utils.generate.note_generate_utils import Note_Generate_Utils
from utils.xc_floder.analysis_xc_note import Analysis_XC_Note


class Note_Generate_Utils_XC(Note_Generate_Utils):
    prompt_template = """参考游记:{text}，你的任务基于'参考游记'，
                创作出一篇全新的游记，在你创作的新游记里，要着重于自然景色的描写，用词尽量不要重复。"""

    def __init__(self,
                 llm,
                 base_file_path: str,
                 max_image_count: int = 9) -> None:
        super().__init__(llm=llm,
                         base_file_path=base_file_path,
                         max_image_count=max_image_count)

    def read_note_list(self, **kwargs):
        try:
            city = kwargs["city"]
            top = int(kwargs["top"])
            is_from_embedding_db = bool(kwargs["is_from_embedding_db"])
            list_note = Analysis_XC_Note.read_note_list(
                city=city, top=top, is_from_embedding_db=is_from_embedding_db)
            return list_note
        except Exception:
            return None

    def summary_response(self, details: list = None) -> str:
        text: str = ""
        # image_urls = []
        if details is not None and len(details) > 0:
            for i in range(0, len(details)):
                cur_text = details[i]["正文"]
                text += f"\n{cur_text}\n"
            return self.generate_article_content(
                text=text, prompt_template=self.prompt_template)
        else:
            return None
