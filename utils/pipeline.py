import queue
from typing import Any, Dict


class Pipeline_Item:
    # 类的实例
    Obj: object = None
    # 方法名
    Method: str = None
    # 方法参数
    Params: Dict[str, Any] = None
    # 是否使用上一个方法的结果作为参数
    Is_Use_Pre_Result: bool = False,
    # 是否在节点执行完之后暂停管道
    Halt: bool = False
    # 当前方法返回结果为None时执行的后备方法
    Standby_Method = None

    def __init__(self,
                 Obj: object,
                 Method: str,
                 Is_Use_Pre_Result: bool = False,
                 Params: Dict[str, Any] = None,
                 Halt: bool = False,
                 Standby_Method=None):
        self.Obj = Obj
        self.Method = Method
        self.Is_Use_Pre_Result = Is_Use_Pre_Result
        self.Params = Params
        self.Halt = Halt
        self.Standby_Method = Standby_Method


class Pipeline_Process_Task:
    pipe_queue: queue.Queue = None
    is_contain_halt_task: bool = False

    def __init__(self) -> None:
        if self.pipe_queue is None:
            self.pipe_queue = queue.Queue()

    def add_item(self, item: Pipeline_Item = None):
        if item is None:
            raise ValueError("添加进管道的执行节点为None")
        if isinstance(item, Pipeline_Item):
            self.pipe_queue.put(item=item)
        else:
            raise ValueError("加进管道的执行节点类型发生错误")

    def execute_pipeline(self, pre_result: Any = None):
        if self.pipe_queue.empty():
            return None
        size = self.pipe_queue.qsize()

        def _excute(item: Pipeline_Item):
            if hasattr(item.Obj, item.Method) and callable(
                    getattr(item.Obj, item.Method)):
                method_to_call = getattr(item.Obj, item.Method)
                # 不使用上一个方法的返回值作为参数
                if item.Is_Use_Pre_Result is not True:
                    if item.Params is not None:
                        result = method_to_call(**item.Params)
                    else:
                        result = method_to_call()
                # 要使用上一个方法的返回值作为参数
                else:
                    if pre_result is not None:
                        if isinstance(pre_result, dict):
                            result = method_to_call(**pre_result)
                        else:
                            result = method_to_call(pre_result)
                    else:
                        result = method_to_call()
                if result is None and item.Standby_Method is not None:
                    result = _excute(item=item.Standby_Method)
                return result

        for i in range(0, size):
            item = self.pipe_queue.get()
            result = _excute(item)
            pre_result = result
            if item.Halt is True:
                self.is_contain_halt_task = True
                break
        return pre_result

    def continue_pipeline(self, pre_result: Any = None):
        return self.execute_pipeline(pre_result)
