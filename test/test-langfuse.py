from langfuse import observe
import os


# Test without OpenAI to verify Langfuse connection
@observe()
def test_function():
    print("Testing Langfuse connection...")
    return "Hello from Langfuse!"


@observe()
def main():
    result = test_function()
    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    print("Starting Langfuse test...")
    main()
    print("Test completed!")
