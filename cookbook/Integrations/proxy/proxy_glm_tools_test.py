import httpx
import asyncio
import json


async def test_zhipu_function_calling():
    ZHIPU_API_KEY = "f0c1fb9f5c534e55a66d9e539916fdb0.GQKa6HaX6MpT9ioJ"
    ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    payload = {
        "model": "glm-4",
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in Berlin? Please use the weather function to check.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a specific city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "The city name"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(
                f"{ZHIPU_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {ZHIPU_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            print("Zhipu Response:")
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # Check if function was called
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            if message.get("tool_calls"):
                print("\n✅ Function calling works!")
            else:
                print(
                    "\n❌ No function calls - GLM might not support it or needs different prompt"
                )

        except Exception as e:
            print(f"Error: {e}")


# Run the test
asyncio.run(test_zhipu_function_calling())
