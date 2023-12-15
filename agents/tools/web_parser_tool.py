import re
from typing import List
from bs4 import BeautifulSoup
import ast
from langchain.document_loaders import AsyncHtmlLoader
from agents.base_tools.base_tools import functional_Tool
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from configs.prompt_config import TOOLS_HTML_TOOL_PROMPT_TEMPLATE
import requests
from configs.base_config import GOOGLE_APIKEY, GOOGLE_SEARCH_ID
import json


class Web_Parser_Tool(functional_Tool):
    llm: BaseLanguageModel
    name = "内容搜索工具"
    description = "基于搜索引擎从互联网搜索相关内容的工具，输入搜索内容关键词，输出对应网页的内容"
    qa_template = TOOLS_HTML_TOOL_PROMPT_TEMPLATE
    prompt = PromptTemplate(input_variables=["query"], template=qa_template)
    llm_chain: LLMChain = None
    base_url = "https://www.googleapis.com/customsearch/v1"
    system_params = f"key={GOOGLE_APIKEY}&cx={GOOGLE_SEARCH_ID}"
    is_parser_detail: bool = True

    def parser_output(self, output: str):
        if not output:
            raise ValueError("未能获取到正确的输入信息")
        # 按换行符分割字符串
        lines = output.split('\n')
        keywords = lines[0].split('：')[1].strip()
        return keywords

    def _call_func(self, query) -> str:
        # 1、提取关键字
        if not query:
            raise ValueError("未能获取到正确的用户信息")
        self.get_llm_chain()
        # 通过用户输入识别出用户输入的关键字，用于到互联网平台搜索内容
        keywords = self.parser_output(self.llm_chain.predict(query=query))
        # searchType=image&start=1
        self.system_params = f"{self.system_params}&q={keywords}&lr=lang_zh-CN&sort=review-rating:d:s"
        search_result = json.loads(
            requests.get(url=f"{self.base_url}?{self.system_params}").text)
        content = []
        if search_result and "items" in search_result:
            if self.is_parser_detail is False:
                for item in search_result["items"]:
                    content.append({
                        "title": item["title"],
                        "link": item["link"],
                        "snippet": item["snippet"]
                    })
            else:
                for item in search_result["items"]:
                    url = item["link"]
                    detail = self._load_html_content(url=url, keyword=keywords)
                    content.append({
                        "title":
                        item["title"],
                        "link":
                        item["link"],
                        "snippet":
                        detail if detail else item["snippet"]
                    })
        return {"sumary_result": {"content": content}}

    def _load_html_content(self, url: str, keyword: str):
        try:
            # 获取响应
            header = {
                "User-Agent":
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            res = requests.get(url=url, headers=header).text
            # 使用正则表达式删除html和script
            content = Web_Parser_Tool._remove_html_and_js(res)
            _prompt = "请从'{content}'中，提炼出与用户输入'{query}'关系最密切的内容"
            prompt_template = PromptTemplate(
                input_variables=["content", "query"], template=_prompt)
            chain = LLMChain(llm=self.llm,
                             prompt=prompt_template,
                             llm_kwargs={
                                 "temperature": 0.95,
                                 "top_p": 0.7
                             })
            content = chain.predict(content=content, query=keyword)
        except:
            content = ""
            pass
        finally:
            return content

    @staticmethod
    def _remove_html_and_js(text):
        text = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '',
                      text)
        text = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '',
                      text)
        text = re.sub(r'<.*?>', '', text)
        text = text.replace("\r", "").replace("\n", "")
        return text

    @staticmethod
    def _remove_js_code(code):
        tree = ast.parse(code, mode='exec')
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.Call)):
                ast.fix_missing_locations(ast.parse('pass').body[0])
                ast.copy_location(ast.parse('pass').body[0], node)
        cleaned_code = ast.unparse(tree)
        return cleaned_code

    def get_llm_chain(self):
        if not self.llm_chain:
            self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
