# 向量数据库
import json
import re
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
# 文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
# 文档加载器 word,pdf,txt,excel
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.document_loaders import DirectoryLoader
# 数据列表类型
from typing import Any, Dict, List
from embeddings.embedding_model import EmbeddingLocalModel
from configs.base_config import BASE_CONFIG


# 操作结果类
class OptResult:

    def __init__(self, isSuccess, msg, contents: List[str]):
        self.isSuccess = isSuccess
        self.msg = msg
        self.contents = contents


class EmbeddingHelper:

    def __init__(
            self,
            persist_directory: str = BASE_CONFIG["chromadb_path"],
            collection_name: str = "ZhoYu",
            local_model_path: str = BASE_CONFIG["embedding_model_path"]
    ) -> None:
        _instance = EmbeddingLocalModel.instance(
            persist_directory=persist_directory,
            collection_name=collection_name,
            local_model_path=local_model_path)
        self.db = _instance.db
        self.embeddingModel = _instance.embeddingModel
        self.persist_directory = persist_directory
        self.collection_name = collection_name

    @classmethod
    def loadDirectory(cls, dic_path: str):
        loader = DirectoryLoader(path=dic_path)
        docs = loader.load()
        return docs

    # 加载文件
    @classmethod
    def loadfile(cls, filePath: str):
        if filePath.find(".docx") != -1:
            loader = Docx2txtLoader(filePath)
            pages = loader.load()
            return pages
        elif filePath.find(".pdf") != -1:
            loader = PyPDFLoader(filePath)
            pages = loader.load()
            return pages
        else:
            loader = TextLoader(filePath)
            pages = loader.load()
            return pages

    # 分割文件
    @classmethod
    def splitDocs(cls, data, chunk_size: int = 200, voerlap: int = 50):
        if not data:
            raise ImportError("没有传入相应的文档数据")
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "。", "......", "！", "？", "?", "!", "."],
                chunk_size=chunk_size,
                chunk_overlap=voerlap)
            docs = text_splitter.split_documents(data)
            return docs

    @classmethod
    def splitText(cls, text: str, chunk_size: int = 200, overlap: int = 50):
        if not text:
            raise ImportError("没有传入相应的文本数据")
        text_spliter = CharacterTextSplitter(chunk_size=chunk_size,
                                             chunk_overlap=overlap)
        text_string = text_spliter.split_text(text)
        text_string = [re.sub(r'\\[ntr]', '', item) for item in text_string]
        return text_string

    @classmethod
    def splitArray(cls,
                   array: List[dict[str, Any]] = None,
                   chunk_size: int = 5) -> List[Document]:
        if array is None or len(array) == 0:
            raise ImportError("没有传入相应的数组数据")

        split_array = [
            array[i:i + chunk_size] for i in range(0, len(array), 5)
        ]
        docs: List[Document] = []
        for group__orignal_array in split_array:
            # texts, metadatas = [], []
            text = json.dumps(group__orignal_array)
            metadata = {"source": group__orignal_array}
            doc = Document(page_content=text, metadata=metadata)
            docs.append(doc)
        return docs

    # 将文档转换成向量并存入向量数据库
    def embedding_docs(self, docs):
        if not docs:
            raise ImportError("没有传入对应的文档切分数据")
        else:
            vector_db = Chroma.from_documents(
                docs,
                embedding=self.embeddingModel,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name)
            return vector_db

    def embedding_texts(self, texts: List[str], metadatas: List[dict]):
        if not texts:
            raise ImportError("没有传入对应的文本数据")
        else:
            vector_db = Chroma.from_texts(
                texts,
                embedding=self.embeddingModel,
                metadatas=metadatas,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name)
            return vector_db

    # 查询最相近的向量
    def query(self,
              message,
              count,
              is_find_metedata: bool = False,
              filter: Dict[str, str] = None,
              where_document: Dict[str, str] = None) -> List[str]:
        if self.db:
            list = []
            result = self.db.similarity_search(message,
                                               count,
                                               filter=filter,
                                               where_document=where_document)
            if result:
                for doc in result:
                    list.append(doc.page_content if is_find_metedata is
                                False else doc.metadata)
            return list
        else:
            raise ImportError("未初始化向量数据库")

    # 文档转向量
    def begin_embedding(self,
                        filepath: str,
                        chunk_size: int = 200,
                        overlap: int = 50) -> OptResult:
        result = OptResult(False, "", [])
        if not filepath:
            result.msg = "未能获取到对应文件的路径，请确保传入文件路径值不为空"
            return result
        pages = self.loadfile(filepath)
        if len(pages) <= 0:
            result.msg = "加载文件时出现错误，未能成功加载到文件内容信息"
            return result
        docs = self.splitDocs(pages, chunk_size, overlap)
        if len(docs) <= 0:
            result.msg = "切分文件时出现错误，未能成功切分到文件内容信息"
            return result
        self.embedding_docs(docs)
        for doc in docs:
            result.contents.append(doc.page_content)
        result.isSuccess = True
        return result
