import time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MODEL_MAP = {
    "gpt-4o": "glm-4",  # Changed to more common GLM-4 model name
}
ZHIPU_API_KEY = "f0c1fb9f5c534e55a66d9e539916fdb0.GQKa6HaX6MpT9ioJ"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


# Fix: Support both /v4/responses and /v1/chat/completions endpoints
@app.api_route("/v1/chat/completions", methods=["POST"])
@app.api_route("/v4/responses/{subpath:path}", methods=["POST"])
@app.post("/v4/responses")
async def responses_proxy(request: Request, subpath: str = ""):
    try:
        body = await request.json()
    except Exception:
        body = {}

    print("📥 Received request:", body)

    # Map model
    requested_model = body.get("model", "gpt-4o")
    mapped_model = MODEL_MAP.get(requested_model, "glm-4")

    # Handle different input formats
    messages = []

    # Check if it's OpenAI Agents format (with input field)
    if "input" in body:
        input_list = body.get("input", [])
        if isinstance(input_list, list) and len(input_list) > 0:
            user_message = input_list[0].get("content", "")
        else:
            user_message = str(input_list)
        messages = [{"role": "user", "content": user_message}]
    # Check if it's standard OpenAI format (with messages field)
    elif "messages" in body:
        messages = body.get("messages", [])
    else:
        # Fallback
        messages = [{"role": "user", "content": "Hello"}]

    print("📥 Messages:", messages)

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
                }
            )
    elif "tools" in body:
        for t in body.get("tools", []):
            if "function" not in t:
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", t.get("schema", {})),
                        },
                    }
                )
            else:
                tools.append(t)

    # Build payload for Zhipu
    payload = {
        "model": mapped_model,
        "messages": messages,
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p", 0.95),
        "max_tokens": body.get("max_output_tokens", body.get("max_tokens", 1024)),
        "stream": body.get("stream", False),
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = body.get("tool_choice", "auto")

    print("📤 Sending to Zhipu:", payload)

    # Call Zhipu API
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{ZHIPU_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {ZHIPU_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            print("📥 Zhipu response:", data)
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, "text") else str(e)
            print(f"❌ Zhipu API error: {e.response.status_code} - {error_detail}")
            return JSONResponse(
                status_code=e.response.status_code,
                content={"error": "智谱请求失败", "detail": error_detail},
            )
        except Exception as e:
            print(f"❌ Proxy error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "代理内部异常", "detail": str(e)},
            )

    # Check if request was for OpenAI Agents (Responses API) or standard OpenAI format
    is_responses_api = "/v4/responses" in str(request.url) or "input" in body

    if is_responses_api:
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
            content.append({"type": "output_text", "text": "No response content"})

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
            "id": data.get("id", f"resp_{int(time.time())}"),
            "object": "response",
            "created": int(time.time()),
            "model": requested_model,  # Return original requested model
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

        print("📤 Responses API response:", resp)
        return JSONResponse(resp)
    else:
        # Return standard OpenAI Chat Completions format
        print("📤 Standard OpenAI response:", data)
        return JSONResponse(data)
