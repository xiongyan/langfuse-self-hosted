import os
from openai import OpenAI


class GLMModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"]
        )

    def get_response(self, prompt: str) -> str:
        print("GLMModel.get_response called with:", prompt)  # <-- 验证点
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content

    async def stream_response(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True,
        )
        async for message in response:
            yield message.choices[0].message.content
