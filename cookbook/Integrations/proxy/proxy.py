import time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MODEL_MAP = {
    "gpt-4o": "GLM-4.5",  # 模型替换
}
ZHIPU_API_KEY = "f0c1fb9f5c534e55a66d9e539916fdb0.GQKa6HaX6MpT9ioJ"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


# -----------------------------
# /v4/responses 代理
# -----------------------------
@app.api_route("/v4/responses/{subpath:path}", methods=["POST"])
@app.post("/v4/responses")
async def responses_proxy(request: Request, subpath: str = ""):
    try:
        body = await request.json()
    except Exception:
        body = {}

    # 提取用户输入
    input_list = body.get("input", [])
    if isinstance(input_list, list) and len(input_list) > 0:
        user_message = input_list[0].get("content", "")
    else:
        user_message = str(input_list)

    # 模型映射
    requested_model = body.get("model", "gpt-4o")
    mapped_model = MODEL_MAP.get(requested_model, "GLM-4.5")

    # 请求智谱 API
    payload = {
        "model": mapped_model,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p", 0.95),
        # "max_output_tokens": body.get("max_output_tokens", 1024),
    }
    print("📤 转发到智谱:", payload)

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            r = await client.post(
                f"{ZHIPU_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {ZHIPU_API_KEY}"},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            return JSONResponse(
                status_code=e.response.status_code,
                content={"error": "智谱请求失败", "detail": e.response.text},
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "代理内部异常", "detail": str(e)},
            )

    # 解析返回内容
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = str(data)

    print("📤 text:", text)

    provider_usage = data.get("usage") or {}

    def _int(v):
        try:
            return int(v or 0)
        except Exception:
            return 0

    input_tokens = _int(
        provider_usage.get("input_tokens")
        or provider_usage.get("input")
        or provider_usage.get("prompt_tokens")
    )
    output_tokens = _int(
        provider_usage.get("output_tokens")
        or provider_usage.get("output")
        or provider_usage.get("completion_tokens")
    )
    total_tokens = _int(
        provider_usage.get("total_tokens")
        or provider_usage.get("total")
        or (input_tokens + output_tokens)
    )

    # -----------------------------
    # ✅ 构造符合 OpenAI Responses API v4 schema 的返回
    # -----------------------------
    resp = {
        "id": data.get("id", "resp_abc123"),
        "object": "response",
        "created": int(time.time()),
        "model": mapped_model,
        "status": "completed",
        "output": [
            {
                "id": data.get("choices", [{}])[0].get("id", "msg_1"),
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_tokens_details": {  # 👈 必须要有
                "cached_tokens": 0,
                "audio_tokens": 0,
            },
            "output_tokens_details": {  # 👈 必须要有
                "reasoning_tokens": 0,
                "audio_tokens": 0,
            },
        },
    }

    print("📤 resp:", resp)
    return JSONResponse(resp)
