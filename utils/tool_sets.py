import importlib
import os
import json
import ast
import shutil


class Tool_Sets:
    class_path: str = "agents\\tools"
    base_module_name = "agents.tools"

    @staticmethod
    def _check_tool(content: str):
        # 解析抽象语法树
        tree = ast.parse(content)
        # 初始化变量，用于记录是否找到指定的类和属性
        found_class = False
        found_name = False
        found_description = False
        found_call_func = False
        name = "",
        description = "",
        classname = ""
        # 遍历抽象语法树
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 判断类是否继承于指定的基类
                base_classes = [base.id for base in node.bases]
                if "functional_Tool" in base_classes:
                    found_class = True
                # 获取工具类名
                classname = node.name
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'name':
                        name = node.value.value
                        found_name = True
                    elif isinstance(target,
                                    ast.Name) and target.id == 'description':
                        description = node.value.value
                        found_description = True

            elif isinstance(node, ast.FunctionDef):
                # 判断是否存在名为 _call_func 的方法
                if node.name == '_call_func':
                    found_call_func = True
        if found_class is False:
            raise ValueError("上传工具未继承于functional_Tool类")
        if found_call_func is False:
            raise ValueError("上传工具未实现_call_func方法")
        if found_name is False:
            raise ValueError("上传工具不包含name属性")
        if found_description is False:
            raise ValueError("上传工具不包含description属性")
        return name, description, classname

    @classmethod
    def load_tools(cls):
        # 获取当前脚本所在目录（main_directory）
        current_dir = os.path.dirname(__file__)
        # 向上一级移动两级，得到项目根目录
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        # 构建相对路径
        relative_path = os.path.join(project_root, "configs", 'tools.json')
        config_data = []
        with open(relative_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
        config_data = [item for item in config_data if item['status'] == 1]
        toos = sorted(config_data,
                      key=lambda x: int(x["sorted"]),
                      reverse=True)
        return toos

    @classmethod
    def regist_tool(cls, sorted: int, file_path: str):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        file_name = os.path.basename(file_path)
        file_name, _ex = os.path.splitext(file_name)
        name, description, classname = cls._check_tool(content)
        relative_path = os.path.join(project_root, "configs", 'tools.json')
        data = {
            "name": name,
            "decription": description,
            "sorted": sorted,
            "status": 1,
            "class_name": classname,
            "module_name": f"{cls.base_module_name}.{file_name}"
        }
        name_is_exist = False
        with open(relative_path, 'r+', encoding='utf-8') as file:
            config_data = json.load(file)
            # 检查工具名称是否重复
            for tool in config_data:
                if name == tool["name"]:
                    name_is_exist = True
                    break
            if name_is_exist is False:
                config_data.append(data)
                file.seek(0)
                json.dump(config_data, file, indent=4)
        if name_is_exist is True:
            raise ValueError("工具名称已经存在，请修改后重新上传")
        module_path = os.path.join(project_root, cls.class_path)
        shutil.copy2(file_path, module_path)
        return cls.load_tools()

    @classmethod
    def init_tools(cls, tool: str, **kwargs):
        config_data = cls.load_tools()
        selected_tools = list(
            filter(lambda element: element.get('name') == tool, config_data))
        if len(selected_tools) <= 0:
            raise ValueError("待加载工具集合中并为发现任何工具信息")
        item = selected_tools[0]
        # 类名的字符串
        class_name = item["class_name"]
        # 根据字符串初始化类
        module_name = item["module_name"]
        # 动态导入模块
        module = importlib.import_module(module_name)
        # 获取类对象
        class_obj = getattr(module, class_name)
        tool_instance = class_obj(**kwargs)
        return tool_instance
