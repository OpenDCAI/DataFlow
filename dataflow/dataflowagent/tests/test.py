import os
import logging
from typing import List, Type
from types import SimpleNamespace

# --- 依赖安装 ---
# 请确保已安装必要的库:
# pip install langchain langchain-openai pydantic python-dotenv

# --- 配置 ---
# 推荐使用 .env 文件来管理你的 API 密钥
# 创建一个 .env 文件，并写入以下内容:
# OPENAI_API_KEY="sk-..."
# OPENAI_API_BASE="https://api.openai.com/v1" # 如果你使用代理，请替换成你的代理地址

from dotenv import load_dotenv
load_dotenv()

# --- LangChain 和 Pydantic 的相关导入 ---
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from pydantic import BaseModel, Field

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# --- 1. 定义我们将要绑定的工具 ---

class GetWeatherInput(BaseModel):
    """定义 get_weather 工具的输入参数结构"""
    city: str = Field(description="需要查询天气的城市名称")

@tool(args_schema=GetWeatherInput)
def get_weather(city: str) -> str:
    """
    一个用于查询指定城市天气的虚拟工具。
    在真实场景中，这里会调用一个天气API。
    """
    log.info(f"--- 模拟调用天气工具，城市: {city} ---")
    if city == "北京":
        return "北京今天是晴天，温度25摄氏度。"
    elif city == "上海":
        return "上海今天有小雨，温度22摄氏度。"
    else:
        return f"抱歉，我没有 {city} 的天气信息。"

class SearchWebInput(BaseModel):
    """定义 search_web 工具的输入参数结构"""
    query: str = Field(description="需要在网络上搜索的查询内容")

@tool(args_schema=SearchWebInput)
def search_web(query: str) -> str:
    """
    一个用于在网络上搜索信息的虚拟工具。
    """
    log.info(f"--- 模拟调用网络搜索工具，查询: {query} ---")
    return f"关于 '{query}' 的搜索结果是：LangChain 是一个强大的LLM框架。"


# --- 2. 模拟你的类和方法 ---
# 为了让测试脚本可以独立运行，我们在这里简化了你的类结构。
# 我们不需要完整的 DFState，只需要一个包含必要信息的模拟对象即可。

class LLMCreator:
    def __init__(self):
        self.model_name = "gpt-4o" # 或者 "gpt-3.5-turbo" 等支持工具调用的模型
        self.temperature = 0.0
        self.max_tokens = 1024
        # self.tool_manager = ... # 在这个测试中我们直接定义工具
    
    def get_post_tools(self) -> List:
        """模拟 get_post_tools 方法，返回我们上面定义的工具列表"""
        return [get_weather, search_web]

    def create_llm(self, state: SimpleNamespace, bind_post_tools: bool = False) -> ChatOpenAI:
        """
        这是你提供的 create_llm 方法的简化和适配版本。
        我们用 SimpleNamespace 来模拟 state 对象。
        """
        actual_model = self.model_name or state.request.model
        log.info(f"创建LLM实例，模型: {actual_model}")
        
        llm = ChatOpenAI(
            # openai_api_base 和 openai_api_key 会自动从环境变量中读取
            # 如果你没有设置环境变量，可以在这里手动传入
            openai_api_base="http://123.129.219.111:3000/v1",
            openai_api_key="sk-",
            model_name=actual_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        if bind_post_tools:
            post_tools = self.get_post_tools()
            if post_tools:
                # 这是我们要测试的核心功能
                llm_with_tools = llm.bind_tools(post_tools)
                log.info(f"为LLM绑定了 {len(post_tools)} 个后置工具: {[t.name for t in post_tools]}")
                return llm_with_tools
        return llm


# --- 3. 编写测试主逻辑 ---

def run_test():
    """
    执行测试的函数
    """
    log.info("========== 开始测试 ==========")

    # 准备模拟的 state 对象
    mock_state = SimpleNamespace(
        request=SimpleNamespace(
            # chat_api_url 和 api_key 将从环境变量加载，这里留空
            chat_api_url=None, 
            api_key=None,
            model=None
        )
    )

    # 实例化我们的LLM创建器
    llm_creator = LLMCreator()

    # 创建绑定了工具的LLM实例
    # 将 bind_post_tools 设置为 True
    llm_with_tools = llm_creator.create_llm(mock_state, bind_post_tools=True)

    # 构造一个应该触发 get_weather 工具的提示
    prompt = "帮我查一下北京今天天气怎么样？"
    log.info(f"发送提示到LLM: '{prompt}'")

    # 调用LLM
    response = llm_with_tools.invoke(prompt)

    # --- 4. 检查和断言返回结果 ---
    log.info("接收到LLM的响应，开始分析...")
    print("\n--- 原始响应对象 ---")
    print(response)
    print("---------------------\n")
    
    # 检查响应类型
    from langchain_core.messages import AIMessage
    assert isinstance(response, AIMessage), "响应应该是一个 AIMessage 对象"

    # **核心断言**：检查 tool_calls 属性
    # 如果LLM决定使用工具，tool_calls 列表将不为空
    assert response.tool_calls, "响应中没有包含任何工具调用信息 (tool_calls)，测试失败！"
    
    log.info("✅ 测试通过: 响应中包含了 tool_calls 属性。")

    # 进一步检查 tool_calls 的内容
    tool_call = response.tool_calls[0]
    print("--- 解析出的工具调用信息 ---")
    print(f"工具名称: {tool_call['name']}")
    print(f"工具参数: {tool_call['args']}")
    print("--------------------------\n")

    # 断言调用的工具是否正确
    assert tool_call['name'] == 'get_weather', f"预期的工具是 'get_weather'，但返回的是 '{tool_call['name']}'"
    log.info(f"✅ 测试通过: 调用的工具名称正确 ({tool_call['name']})。")

    # 断言工具的参数是否正确
    expected_args = {'city': '北京'}
    assert tool_call['args'] == expected_args, f"预期的参数是 {expected_args}，但返回的是 {tool_call['args']}"
    log.info(f"✅ 测试通过: 调用的工具参数正确 ({tool_call['args']})。")
    
    # 检查 content 属性是否为空
    # 当进行工具调用时，模型的主要回复就在 tool_calls 中，content 通常是空的
    assert response.content == "", f"预期 content 为空字符串，但返回了 '{response.content}'"
    log.info("✅ 测试通过: 响应的 content 属性为空字符串，符合工具调用预期。")

    log.info("\n========== 所有断言均通过，测试成功！ ==========")
    log.info("这证明了 llm.bind_tools(post_tools) 成功地将工具绑定到了LLM，并且LLM在收到合适的提示时，能够正确地返回工具调用的结构化信息。")


if __name__ == "__main__":
    # 检查 API Key 是否设置
    if not os.getenv("OPENAI_API_KEY"):
        print("\n错误：请先设置 OPENAI_API_KEY 环境变量。")
        print("你可以创建一个 .env 文件，并写入 OPENAI_API_KEY=\"your_key_here\"。\n")
    else:
        run_test()