import os
from dataflow.serving import APILLMServing_request  # 根据你的实际路径修改


def test_openai_serving():
    # 检查 API_KEY 是否存在
    if "API_KEY" not in os.environ:
        raise RuntimeError("请先设置环境变量 API_KEY，例如：export API_KEY='sk-xxx'")

    # 初始化 Serving
    serving = APILLMServing_request(
        # api_url="https://api.openai.com/v1/chat/completions",
        api_url="http://123.129.219.111:3000/v1/chat/completions",
        model_name="gpt-4o"
    )

    # 多轮对话输入（格式为 list[list[dict]]）
    conversations = [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who wrote Hamlet?"},
        {"role": "assistant", "content": "Hamlet was written by William Shakespeare."},
        {"role": "user", "content": "When was it written?"}
    ]]

    # 调用接口
    responses = serving.generate_from_conversations(conversations)

    # 打印结果
    print("输入对话：")
    for turn in conversations[0]:
        print(f"{turn['role']}: {turn['content']}")
    print("\n模型输出：")
    print(responses[0])

    # 简单检查
    if responses[0] is None:
        raise RuntimeError("请求失败，返回 None")
    # elif "1600" not in responses[0] and "17" not in responses[0]:
    #     raise RuntimeError(f"模型返回可能不包含预期年份信息：{responses[0]}")

    print("✅ 多轮对话 API 调用测试成功")

# 运行测试
if __name__ == "__main__":
    test_openai_serving()
