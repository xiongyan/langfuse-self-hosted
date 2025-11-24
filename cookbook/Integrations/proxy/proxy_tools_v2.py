import time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MODEL_MAP = {
    "gpt-4o": "glm-4.5",  # 模型替换 (note: GLM-4.5 may not be exact name; check Zhipu docs, often "glm-4" or "glm-4-airx")
}
ZHIPU_API_KEY = "f0c1fb9f5c534e55a66d9e539916fdb0.GQKa6HaX6MpT9ioJ"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


@app.api_route("/v4/responses/{subpath:path}", methods=["POST"])
@app.post("/v4/responses")
async def responses_proxy(request: Request, subpath: str = ""):
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Map model
    requested_model = body.get("model", "gpt-4o")
    mapped_model = MODEL_MAP.get(requested_model, "glm-4.5")

    # Build messages from input (handle multi-part content)
    input_list = body.get("input", [])
    if isinstance(input_list, list) and len(input_list) > 0:
        user_message = input_list[0].get("content", "")
    else:
        user_message = str(input_list)

    print("user_message: ", user_message)

    # Handle tools/functions conversion
    tools = []
    if "functions" in body and not body.get("tools"):
        for f in body.get("functions", []):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f.get("name"),
                        "description": f.get("description", ""),
                        "parameters": f.get("parameters", f.get("schema", {})),
                    },
                    **({k: f[k] for k in ("strict",) if k in f}),
                }
            )
    elif "tools" in body:
        for t in body.get("tools", []):
            if "function" not in t:
                tools.append(
                    {
                        **t,
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", t.get("schema", {})),
                        },
                        "type": t.get("type", "function"),
                    }
                )
            else:
                tools.append(t)

    # Build payload for Zhipu
    payload = {
        "model": mapped_model,
        "messages": [{"role": "user", "content": user_message}],
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p", 0.95),
        "max_tokens": body.get("max_output_tokens", 1024),
    }
    if tools:
        payload["tools"] = tools

    print("\n")
    print("tools:", tools)
    print("\n📤 转发到智谱:", payload)

    # Call Zhipu API
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

    # Parse Zhipu response and convert to Responses API format
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = []
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            content.append(
                {
                    "type": "tool_call",
                    "tool_call": {
                        "id": tc.get("id", f"call_{int(time.time())}"),
                        "type": tc.get("type", "function"),
                        "function": tc.get("function", {}),
                    },
                }
            )
    elif message.get("content"):
        content.append({"type": "output_text", "text": message["content"]})
    else:
        content.append({"type": "output_text", "text": str(data)})  # Fallback

    print("📤 content:", content)

    # Usage mapping
    provider_usage = data.get("usage", {})

    def _int(v):
        try:
            return int(v or 0)
        except Exception:
            return 0

    input_tokens = _int(provider_usage.get("prompt_tokens"))
    output_tokens = _int(provider_usage.get("completion_tokens"))
    total_tokens = _int(
        provider_usage.get("total_tokens") or (input_tokens + output_tokens)
    )

    # Build Responses API response
    resp = {
        "id": data.get("id", "resp_abc123"),
        "object": "response",
        "created": int(time.time()),
        "model": mapped_model,
        "status": "completed",
        "output": [
            {
                "id": choice.get("index", 0),
                "type": "message",
                "role": "assistant",
                "content": content,
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0},
        },
    }

    print("📤 resp:", resp)
    return JSONResponse(resp)
