import os

# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-b26f6b38-1d21-4efb-b0ed-5057200b2148"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-842c4b9d-4e56-4325-9df5-5a09408f42b1"

LANGFUSE_HOST = "http://localhost:3000"

# Your openai key
# https://api.siliconflow.cn
# os.environ["OPENAI_API_KEY"] = "sk-nfyvlkiykssllzekmddumkfxggollxgwiteckpcqaxchtgty"
os.environ["OPENAI_API_KEY"] = "73c80b33ad68446ea3f059efe5c1a65f.T2PZjYiHcT2JYx2a"
os.environ["OPENAI_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"


from langfuse.openai import openai

client = openai.OpenAI()

system_prompt = (
    "You are a very accurate calculator. You output only the result of the calculation."
)

completions = client.chat.completions.create(
    model="glm-4.5-flash",  # 可选择其他的模型
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "2 + 2 = "},
    ],
)

print(completions.choices[0].message.content)


# Summarize cost by model
import pandas as pd
from langfuse import observe, get_client

langfuse = get_client()
trace = langfuse.api.trace.get("22ce3982b715aa79c077c6443554934d")
observations = trace.observations


def summarize_usage(observations):
    """Summarize usage data grouped by model."""
    usage_data = []

    for obs in observations:
        usage = obs.usage
        if usage:
            usage_data.append(
                {
                    "model": obs.model,
                    "input_tokens": usage.input,
                    "output_tokens": usage.output,
                    "total_tokens": usage.total,
                }
            )

    df = pd.DataFrame(usage_data)
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby("model").sum()
    return summary


# Example usage (assuming `observations` is defined as in the provided code):
summary_df = summarize_usage(observations)
summary_df
