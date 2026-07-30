import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com"
    )

def sentiment_analysis(text: str) -> str:
    """
    Zero-shot 情感分析：直接让模型判断
    """
    prompt = f"""请判断以下评论的情感倾向，只回复"正面"或"负面"或"中性"，不要解释。

    评论：{text}
    情感："""
    
    response = client.chat.completions.create(
        model="deepseek-chat",  # 或 "deepseek-reasoner"（推理模型）
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0  # 0 表示最确定，不要随机发挥
    )
    
    return response.choices[0].message.content.strip()


def sentiment_analysis_fewshot(text: str) -> str:
    """
    Few-shot: 给例子, 让模型学你的格式
    """
    prompt = """判断评论情感，只回复"正面"或"负面"。

    评论：这部电影太棒了，演员演技在线！
    情感：正面

    评论：排队两小时，味道一般，不值这个价。
    情感：负面

    评论：包装破损，客服态度还差，差评。
    情感：负面

    评论：{text}
    情感：""".format(text=text)
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()


# 测试
if __name__ == "__main__":
    texts = [
        "这家餐厅太难吃了，服务员态度还差。",
        "今天天气不错，心情很好。",
        "快递等了三天，结果东西还是坏的。"
    ]
    
    print("=== Zero-shot ===")
    for t in texts:
        print(f"{t[:15]}... -> {sentiment_analysis(t)}")
    
    print("\n=== Few-shot ===")
    for t in texts:
        print(f"{t[:15]}... -> {sentiment_analysis_fewshot(t)}")

