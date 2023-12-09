SEQUENCE_EXECUTE_API_TOOLS_PROMPT_TEMPLATE: str = """
    已知工具信息：{intents}，你需要根据已知工具信息，自主思考为了满足用户意图，必须从已知工具中按从前到后的逻辑顺序调用哪一个或哪几个工具（某些工具的输出可以作为另一些工具的输入），
    并且你的输出格式必须按照：
    工具调用链：[你认为要调用的工具的名称。如果需要调用多个工具，请用按从前到后的调用顺序输出工具名称，并用逗号分割]
    不能随意更改输出格式，如果已知工具中没有任何一个工具可以满足用户意图，请直接输出'[未知工具]'

    例如：
    
    用户输入：帮我生成一篇以敦煌为主题的旅行游记
    工具调用链：[旅行游记生成工具]

    用户输入：帮我画一幅画
    工具调用链：[未知工具]

    用户输入：'{query}'
    """
SINGLE_EXECUTE_API_TOOLS_PROMPT_TEMPLATE: str = """
    现在有一些意图，类别为{intents}，你的任务是理解用户问题的意图，并判断该问题属于哪一类意图。
    回复的意图类别必须在提供的类别中，并且必须按格式回复：“意图类别：<>”。
    
    举例：
    问题：中海国际36亩地块什么时候立项的？
    意图类别：查询项目基本信息
    
    问题：葛洲坝项目的人员策划哪些专业已经完成了？
    意图类别：查询项目人员信息
    
    问题：安川2号地块项目建筑专业的预估工时是多少？
    意图类别：查询项目工时信息

    问题：“{query}”
    """

TOOLS_NOTE_TOOL_PROMPT_TEMPLATE: str = """
    你的任务是从用户的输入信息中提取出用于互联网搜索的关键字。
    必须按格式回复：“关键字：<>”。
    
    举例：
    问题：国庆节去大西北应该怎么玩？
    关键字：大西北
    
    问题：成都当地有什么特色小吃？
    关键字：成都

    问题：“{query}”
    """
TOOLS_CONTENT_TOOL_PROMPT_TEMPLATE: str = """
    你的任务是从用户的输入信息中提取出关键词。
    你的输出格式必须按照：

    关键词：你提取出的关键词

    这样的格式进行输出。不能额外添加任何输出内容，必须严格遵守标准格式进行输出

    例如：

    用户输入：请帮我生成一篇以屈原的离骚为主题的文学作品介绍
    关键词：离骚
    
    用户输入：{query}
    """

TOOLS_HTML_TOOL_PROMPT_TEMPLATE: str = """
    你的任务是从用户的输入信息中提取出关键信息。
    你的输出格式必须按照：

    关键信息：你从用户输入中提取出的用于搜索引擎搜索的关键信息

    这样的格式进行输出。不能额外添加任何输出内容，必须严格遵守标准格式进行输出

    例如：

    用户输入：这两天成都有什么新闻？
    关键信息：成都，新闻
    
    用户输入：{query}
    """