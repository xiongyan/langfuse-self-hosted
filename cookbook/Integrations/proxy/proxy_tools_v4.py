# ...existing code...
import os
import json
import time
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

MODEL_MAP = {
    "gpt-4o": "glm-4.5",  # Try GLM-4 which might have better function support
    # Alternative models to try:
    # "gpt-4o": "glm-4-air",
    # "gpt-4o": "glm-4-airx",
    # "gpt-4o": "glm-4-flash",
}
ZHIPU_API_KEY = "f0c1fb9f5c534e55a66d9e539916fdb0.GQKa6HaX6MpT9ioJ"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def normalize_messages_for_upstream(messages):
    """
    Ensure each message['content'] is a string.
    Convert structured fragments (tool_call/tool_result/output_text) into readable text.
    """
    normalized = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # If content is a list of fragments, flatten it
        if isinstance(content, list):
            parts = []
            for frag in content:
                if isinstance(frag, dict):
                    t = frag.get("type", "")
                    # fragments that represent tool calls/results
                    if t in ("tool_call", "tool_result") or frag.get("tool_call"):
                        tc = frag.get("tool_call") or frag
                        name = (
                            (tc.get("function") or tc).get("name")
                            if isinstance(
                                tc.get("function") if isinstance(tc, dict) else None,
                                dict,
                            )
                            else tc.get("name") or tc.get("tool")
                        )
                        if isinstance(tc.get("arguments"), (dict, list)):
                            args_text = json.dumps(
                                tc.get("arguments"), ensure_ascii=False
                            )
                        else:
                            args_text = str(tc.get("arguments", ""))
                        parts.append(f"[tool_call:{name} {args_text}]")
                    elif t in ("text", "output_text"):
                        parts.append(str(frag.get("text") or frag.get("content") or ""))
                    else:
                        parts.append(json.dumps(frag, ensure_ascii=False))
                else:
                    parts.append(str(frag))
            content_str = "".join(parts)
        elif isinstance(content, dict):
            # convert dict content to string
            content_str = json.dumps(content, ensure_ascii=False)
        else:
            content_str = str(content or "")
        normalized.append(
            {
                "role": role,
                "content": content_str,
                **({k: v for k, v in m.items() if k not in ("role", "content")}),
            }
        )
    return normalized


@app.api_route("/v4/responses/{subpath:path}", methods=["POST"])
@app.api_route("/v4/responses", methods=["POST"])
async def responses_proxy(request: Request, subpath: str = ""):
    try:
        body = await request.json()
        print(">>> Incoming request to proxy:", json.dumps(body, ensure_ascii=False))
    except Exception:
        body = {}

    print("📥 Received request body keys:", list(body.keys()))
    # print("📥 Full request body:", body)

    # Map model
    requested_model = body.get("model", "gpt-4o")
    mapped_model = MODEL_MAP.get(requested_model, "glm-4.5")

    # Handle different input formats (build `messages` from body["input"] or body["messages"])
    messages = []
    if "input" in body:
        input_list = body.get("input", [])
        if isinstance(input_list, list):
            for inp in input_list:
                if isinstance(inp, dict):
                    role = inp.get("role") or inp.get("from") or "user"
                    content = inp.get("content", "")
                    messages.append({"role": role, "content": content})
                else:
                    messages.append({"role": "user", "content": str(inp)})
        else:
            messages.append({"role": "user", "content": str(input_list)})
    elif "messages" in body:
        messages = body.get("messages", [])
    else:
        # no input -> keep empty or return 400 earlier
        messages = []

    # Handle different input formats
    messages = normalize_messages_for_upstream(messages)

    print("📥 Messages:", messages)

    # Handle tools/functions conversion with better descriptions
    tools = []
    if "functions" in body and not body.get("tools"):
        for f in body.get("functions", []):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f.get("name"),
                        "description": f.get(
                            "description", f"Call the {f.get('name')} function"
                        ),
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
                            "description": t.get(
                                "description", f"Call the {t.get('name')} function"
                            ),
                            "parameters": t.get("parameters", t.get("schema", {})),
                        },
                    }
                )
            else:
                # Enhance existing function descriptions if empty
                func = t.get("function", {})
                if not func.get("description"):
                    func["description"] = (
                        f"Call the {func.get('name')} function to get information"
                    )
                tools.append(t)

    # Build payload for Zhipu
    payload = {
        "model": mapped_model,
        "messages": messages,
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p", 0.95),
        # "max_tokens": body.get("max_output_tokens", body.get("max_tokens", 1024)),
        "stream": body.get("stream", False),
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "required"  # Force function calling instead of "auto"

        # Add system message to encourage function usage for GLM
        if not any(msg.get("role") == "system" for msg in messages):
            system_msg = {
                "role": "system",
                "content": "You are a helpful assistant. When the user asks for information that requires using available functions/tools, you should call the appropriate function first before responding.",
            }
            payload["messages"] = [system_msg] + messages

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

    # --- BEGIN: Responses API normalization + optional local tool execution (temporary) ---
    if is_responses_api:
        import json as _json
        import time

        # Simple local tool registry for quick debugging.
        # Add more functions here if you want proxy to execute them locally.
        def _local_get_weather(args):
            city = args.get("city") if isinstance(args, dict) else None
            return f"The weather in {city or 'unknown'} is sunny"

        LOCAL_TOOL_REGISTRY = {
            "get_weather": _local_get_weather,
            # "other_tool": callable_here,
        }

        choice = data.get("choices", [{}])[0]
        message = choice.get("message") or {}

        # collect tool_calls from common locations
        downstream = choice.get("tool_calls") or message.get("tool_calls") or []
        if not downstream:
            content = message.get("content") or []
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_call":
                        tc = c.get("tool_call") or c
                        downstream.append(tc)

        # normalize and parse arguments
        normalized = []
        for tc in downstream:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or tc.get("arguments") or {}
            args = args_raw
            if isinstance(args_raw, str):
                try:
                    args = _json.loads(args_raw)
                except Exception:
                    args = {"__raw": args_raw}
            name = fn.get("name") or tc.get("name") or None
            cid = tc.get("id") or f"call_{int(time.time()*1000)}"
            normalized.append(
                {
                    "id": str(cid),
                    "name": name,
                    "arguments": args,
                    "type": tc.get("type", "function"),
                }
            )

        outputs = []

        # 1) top-level tool_call entries (Agents expects these)
        for nc in normalized:
            outputs.append(
                {
                    "id": str(nc["id"]),
                    "type": "tool_call",
                    "tool_call": {
                        "id": str(nc["id"]),
                        "type": nc["type"],
                        "name": nc["name"],
                        "arguments": nc["arguments"],
                        "function": {"name": nc["name"], "arguments": nc["arguments"]},
                    },
                }
            )

        # 2) If possible, execute local tool and inject tool-result + synthesize assistant reply
        for nc in normalized:
            tool_name = nc["name"]
            tool_args = nc["arguments"] or {}
            tool_result_text = None
            if tool_name and tool_name in LOCAL_TOOL_REGISTRY:
                try:
                    tool_result_text = LOCAL_TOOL_REGISTRY[tool_name](tool_args)
                    # inject a tool message (role="tool") with name + content
                    outputs.append(
                        {
                            "id": f"tool_{nc['id']}",
                            "type": "message",
                            "role": "tool",
                            "name": tool_name,
                            "content": [
                                {"type": "output_text", "text": tool_result_text}
                            ],
                        }
                    )
                except Exception as e:
                    tool_result_text = f"__tool_error__: {e}"
                    outputs.append(
                        {
                            "id": f"tool_err_{nc['id']}",
                            "type": "message",
                            "role": "tool",
                            "name": tool_name,
                            "content": [
                                {"type": "output_text", "text": tool_result_text}
                            ],
                        }
                    )

        # 3) assistant message content: only output_text (do NOT embed tool_call here)
        text = ""
        if isinstance(message.get("content"), str) and message.get("content"):
            text = message.get("content")
        elif isinstance(choice.get("text"), str) and choice.get("text"):
            text = choice.get("text")

        # If we executed any local tool above, prefer its synthesized result as assistant text
        if normalized:
            # prefer last tool result if present
            last_tool_msg = next(
                (o for o in reversed(outputs) if o.get("role") == "tool"), None
            )
            if last_tool_msg:
                text = last_tool_msg.get("content", [{}])[0].get("text", "") or text

        content_list = [{"type": "output_text", "text": text or ""}]

        outputs.append(
            {
                "id": str(choice.get("index", "0")),
                "type": "message",
                "role": "assistant",
                "content": content_list,
            }
        )

        # usage mapping with safe defaults for details (must be dicts)
        provider_usage = data.get("usage") or {}

        def _int(v):
            try:
                return int(v or 0)
            except Exception:
                return 0

        input_tokens = _int(
            provider_usage.get("prompt_tokens")
            or provider_usage.get("input_tokens")
            or provider_usage.get("input")
        )
        output_tokens = _int(
            provider_usage.get("completion_tokens")
            or provider_usage.get("output_tokens")
            or provider_usage.get("output")
        )
        total_tokens = _int(
            provider_usage.get("total_tokens")
            or provider_usage.get("total")
            or (input_tokens + output_tokens)
        )

        input_tokens_details = provider_usage.get("input_tokens_details") or {
            "cached_tokens": 0,
            "audio_tokens": 0,
        }
        output_tokens_details = provider_usage.get("output_tokens_details") or {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
        }

        resp = {
            "id": data.get("id", f"resp_{int(time.time())}"),
            "object": "response",
            "created": int(time.time()),
            "model": requested_model,
            "status": "completed",
            "output": outputs,
            "outputs": outputs,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_tokens_details": input_tokens_details,
                "output_tokens_details": output_tokens_details,
            },
        }

        print("📤 Responses API response (normalized + local-exec):", resp)
        return JSONResponse(resp)
    # --- END: Responses API normalization + optional local tool execution ---

    else:
        # Return standard OpenAI Chat Completions format
        print("📤 Standard OpenAI response:", data)
        return JSONResponse(data)
