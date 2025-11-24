import os
import time
import copy
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("proxy")

app = FastAPI()

MODEL_MAP = {
    "gpt-4o": "GLM-4.5",
    # add other mappings as needed
}

ZHIPU_API_KEY = "f0c1fb9f5c534e55a66d9e539916fdb0.GQKa6HaX6MpT9ioJ"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def _int(v: Any) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0


def _ensure_details(d: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not isinstance(d, dict):
        d = {}
    return {
        "prompt_tokens": _int(d.get("prompt_tokens")),
        "reasoning_tokens": _int(d.get("reasoning_tokens")),
        "other_tokens": _int(d.get("other_tokens")),
        "completion_tokens": _int(d.get("completion_tokens")),
        "tool_calls_tokens": _int(d.get("tool_calls_tokens") or d.get("tool_calls")),
        "final_tokens": _int(d.get("final_tokens")),
        "planning_tokens": _int(d.get("planning_tokens")),
    }


def _normalize_functions_to_tools(
    functions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    tools = []
    for f in functions:
        tools.append(
            {
                "name": f.get("name"),
                "type": "function",
                "description": f.get("description", "") or "",
                "function": {
                    "name": f.get("name"),
                    "description": f.get("description", "") or "",
                    "parameters": f.get("parameters") or f.get("schema") or {},
                },
                **({k: f[k] for k in ("strict",) if k in f}),
            }
        )
    return tools


async def _forward_to_zhipu(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {}
    if ZHIPU_API_KEY:
        headers["Authorization"] = f"Bearer {ZHIPU_API_KEY}"
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{ZHIPU_BASE_URL}/chat/completions", json=payload, headers=headers
        )
        r.raise_for_status()
        return r.json()


def _ensure_messages_from_input(
    body: Dict[str, Any], payload: Dict[str, Any], user_message: str
) -> None:
    if "messages" in payload:
        return
    inp = payload.get("input", body.get("input"))
    if isinstance(inp, list) and len(inp) > 0:
        msgs = []
        for itm in inp:
            if isinstance(itm, dict) and "content" in itm:
                role = itm.get("role", "user")
                msgs.append({"role": role, "content": itm.get("content", "")})
            else:
                msgs.append({"role": "user", "content": str(itm)})
        payload["messages"] = msgs
    elif isinstance(inp, dict) and "content" in inp:
        payload["messages"] = [
            {"role": inp.get("role", "user"), "content": inp.get("content", "")}
        ]
    elif inp is not None:
        payload["messages"] = [{"role": "user", "content": str(inp)}]
    else:
        payload["messages"] = [{"role": "user", "content": user_message}]


def _build_response_shape(data: Dict[str, Any], model: str) -> Dict[str, Any]:
    provider_usage = data.get("usage") or {}
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

    # extract text(s)
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = str(data)

    # ensure details
    input_details = _ensure_details(
        provider_usage.get("input_tokens_details")
        or provider_usage.get("input_details")
        or provider_usage
    )
    output_details = _ensure_details(
        provider_usage.get("output_tokens_details")
        or provider_usage.get("output_details")
        or provider_usage
    )

    resp = {
        "id": data.get("id", ""),
        "object": "response",
        "created": int(time.time()),
        "model": model,
        "status": "completed",
        "output": [
            {
                "id": data.get("choices", [{}])[0].get("id", "msg_1"),
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
                # preserve tool_calls if present to help SDK detect function invocation
                **(
                    {"tool_calls": data.get("choices", [{}])[0].get("tool_calls")}
                    if data.get("choices", [{}])[0].get("tool_calls")
                    else {}
                ),
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            # legacy names
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
            # details (top-level usage.details kept under usage for compatibility)
            "input_tokens_details": input_details,
            "output_tokens_details": output_details,
        },
    }
    return resp


@app.api_route("/v4/responses/{subpath:path}", methods=["POST"])
@app.post("/v4/responses")
@app.post("/responses")
@app.post("/v1/responses")
async def responses_proxy(request: Request, subpath: str = ""):
    try:
        body = await request.json()
    except Exception:
        body = {}

    # extract user_message for fallback
    input_list = body.get("input", [])
    if isinstance(input_list, list) and len(input_list) > 0:
        first = input_list[0]
        user_message = (
            first.get("content", "") if isinstance(first, dict) else str(first)
        )
    else:
        user_message = str(input_list or body.get("messages") or "")

    requested_model = body.get("model", "gpt-4o")
    mapped_model = MODEL_MAP.get(
        requested_model, requested_model if requested_model else "GLM-4.5"
    )

    # prepare payload: deep copy original to preserve functions/tools/function_call
    payload: Dict[str, Any] = copy.deepcopy(body) if isinstance(body, dict) else {}

    # ensure messages exist
    _ensure_messages_from_input(body, payload, user_message)

    # normalize model/token params
    payload["model"] = mapped_model
    if "max_tokens" not in payload and "max_output_tokens" in body:
        payload["max_tokens"] = body["max_output_tokens"]

    # forward function_call if present and not already in payload
    if "function_call" in body and "function_call" not in payload:
        payload["function_call"] = body["function_call"]

    # compatibility: if 'functions' provided, convert to 'tools' expected by Zhipu
    if "functions" in payload and "tools" not in payload:
        try:
            payload["tools"] = _normalize_functions_to_tools(payload["functions"])
        except Exception:
            payload["tools"] = []
    # if tools exist but items lack "function", ensure structure
    if "tools" in payload:
        normalized_tools = []
        for t in payload["tools"] or []:
            if t is None:
                continue
            if isinstance(t, dict) and "function" not in t:
                normalized_tools.append(
                    {
                        **t,
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description", "") or "",
                            "parameters": t.get("parameters") or t.get("schema") or {},
                        },
                        "type": t.get("type", "function"),
                    }
                )
            else:
                normalized_tools.append(t)
        payload["tools"] = normalized_tools

    log.info("Forwarding to Zhipu keys=%s", list(payload.keys()))
    log.debug(
        "Forward payload preview=%s", json.dumps(payload, ensure_ascii=False)[:2000]
    )

    try:
        data = await _forward_to_zhipu(payload)
    except httpx.HTTPStatusError as e:
        detail = None
        try:
            detail = e.response.text
        except Exception:
            detail = str(e)
        log.error("Zhipu downstream error: %s", detail)
        return JSONResponse(
            status_code=e.response.status_code if e.response is not None else 502,
            content={"error": "智谱请求失败", "detail": detail},
        )
    except Exception as e:
        log.exception("Proxy internal error")
        return JSONResponse(
            status_code=500, content={"error": "代理内部异常", "detail": str(e)}
        )

    log.debug("Zhipu response preview=%s", json.dumps(data, ensure_ascii=False)[:2000])

    resp = _build_response_shape(data, mapped_model)
    return JSONResponse(resp)
