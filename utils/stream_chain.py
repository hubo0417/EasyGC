from typing import Any, Mapping, Optional
from langchain.callbacks.base import Callbacks
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import LoadingCallable, map_reduce_prompt
from langchain.load.dump import dumpd
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.prompt_template import BasePromptTemplate
from pydantic.fields import Field
from langchain.callbacks.manager import CallbackManager


class Stream_Chain(LLMChain):
    # 重写Predict方法，使其变成流式响应
    def predict(self, callbacks: Callbacks = None, **kwargs: Any):
        inputs = self.prep_inputs(kwargs)
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            None,
            self.tags,
            None,
            self.metadata,
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            inputs,
            name=None,
        )
        prompts, stop = self.prep_prompts([inputs], run_manager=run_manager)
        prompt_strings = [p.to_string() for p in prompts]
        prompt = "\n".join(prompt_strings)
        stream_generate = self.llm.stream(prompt, stop=stop, **self.llm_kwargs)
        return stream_generate


class Stream_Map_Reduce_Chain(MapReduceDocumentsChain):
    llm: BaseLanguageModel = None


def _load_map_reduce_chain(
    llm: BaseLanguageModel,
    map_prompt: BasePromptTemplate = map_reduce_prompt.PROMPT,
    combine_prompt: BasePromptTemplate = map_reduce_prompt.PROMPT,
    combine_document_variable_name: str = "text",
    map_reduce_document_variable_name: str = "text",
    collapse_prompt: Optional[BasePromptTemplate] = None,
    reduce_llm: Optional[BaseLanguageModel] = None,
    collapse_llm: Optional[BaseLanguageModel] = None,
    verbose: Optional[bool] = None,
    token_max: int = 3000,
    callbacks: Callbacks = None,
    llm_kwargs: dict = Field(default_factory=dict),
    **kwargs: Any,
) -> Stream_Map_Reduce_Chain:
    map_chain = Stream_Chain(llm=llm,
                             prompt=map_prompt,
                             verbose=verbose,
                             callbacks=callbacks,
                             llm_kwargs=llm_kwargs)
    reduce_chain = Stream_Chain(llm=llm,
                                prompt=combine_prompt,
                                verbose=verbose,
                                callbacks=callbacks,
                                llm_kwargs=llm_kwargs)
    # TODO: document prompt
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name=combine_document_variable_name,
        verbose=verbose,
        callbacks=callbacks,
    )
    if collapse_prompt is None:
        collapse_chain = None
        if collapse_llm is not None:
            raise ValueError(
                "collapse_llm provided, but collapse_prompt was not: please "
                "provide one or stop providing collapse_llm.")
    else:
        collapse_chain = StuffDocumentsChain(
            llm_chain=Stream_Chain(
                llm=llm,
                prompt=collapse_prompt,
                verbose=verbose,
                callbacks=callbacks,
            ),
            document_variable_name=combine_document_variable_name,
        )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=collapse_chain,
        token_max=token_max,
        verbose=verbose,
        callbacks=callbacks,
    )
    return Stream_Map_Reduce_Chain(
        llm=llm,
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name=map_reduce_document_variable_name,
        verbose=verbose,
        callbacks=callbacks,
        **kwargs,
    )


def load_summarize_chain(
    llm: BaseLanguageModel,
    chain_type: str = "stuff",
    verbose: Optional[bool] = None,
    llm_kwargs: dict = Field(default_factory=dict),
    **kwargs: Any,
) -> Stream_Map_Reduce_Chain:
    loader_mapping: Mapping[str, LoadingCallable] = {
        "map_reduce": _load_map_reduce_chain
    }
    if chain_type not in loader_mapping:
        raise ValueError(f"Got unsupported chain type: {chain_type}. "
                         f"Should be one of {loader_mapping.keys()}")
    return loader_mapping[chain_type](llm,
                                      verbose=verbose,
                                      llm_kwargs=llm_kwargs,
                                      **kwargs)
