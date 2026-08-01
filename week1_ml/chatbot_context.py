import os
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


class LearningAssistant:
    """
    多轮对话学习助手，维护上下文记忆
    """
    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self.messages = []  # 对话历史
        self._set_system_prompt()
    
    def _set_system_prompt(self):
        """
        System Prompt：设定 AI 的角色和行为模式
        这是控制 LLM 输出的最强手段
        """
        system_prompt = """你是一个专业的 诗人，擅长用文言文解释机器学习概念。
        
规则：
1. 回答不超过 3 句话
2. 遇到数学公式用 LaTeX 格式
3. 如果用户问代码问题，先给核心思路，再给出代码
4. 不知道就承认，不要编造"""
        
        self.messages.append({"role": "system", "content": system_prompt})
    
    def chat(self, user_input: str) -> str:
        """
        发送消息并获取回复，同时维护对话历史
        """
        # 1. 添加用户消息
        self.messages.append({"role": "user", "content": user_input})
        
        # 2. 调用 API（带上完整历史）
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7
        )
        
        # 3. 获取助手回复
        assistant_reply = response.choices[0].message.content
        
        # 4. 把回复加入历史（关键！下次对话会带上上下文）
        self.messages.append({"role": "assistant", "content": assistant_reply})
        
        return assistant_reply
    
    def clear_history(self):
        """清空对话（但保留 system prompt）"""
        self.messages = [self.messages[0]]
    
    def get_history(self) -> list:
        """查看当前对话历史"""
        return self.messages


# ========== 测试 ==========
if __name__ == "__main__":
    assistant = LearningAssistant()
    
    # 第一轮
    q1 = "什么是梯度下降？"
    print(f"用户: {q1}")
    print(f"助手: {assistant.chat(q1)}\n")
    
    # 第二轮（带上下文！助手知道"它"指代梯度下降）
    q2 = "它和牛顿法有什么区别？"
    print(f"用户: {q2}")
    print(f"助手: {assistant.chat(q2)}\n")
    
    # 第三轮（继续深入）
    q3 = "用 Python 写一个最简单的例子"
    print(f"用户: {q3}")
    print(f"助手: {assistant.chat(q3)}\n")
    
    # 查看历史
    print("=== 对话历史 ===")
    for msg in assistant.get_history():
        print(f"{msg['role']}: {msg['content'][:50]}...")