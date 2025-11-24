import os

print("Environment variables:")
print(f"LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST')}")
print(f"LANGFUSE_PUBLIC_KEY: {os.getenv('LANGFUSE_PUBLIC_KEY')}")
print(f"LANGFUSE_SECRET_KEY: {os.getenv('LANGFUSE_SECRET_KEY')}")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")

print("\nTesting basic imports...")
try:
    from langfuse import observe

    print("✓ langfuse.observe imported successfully")
except Exception as e:
    print(f"✗ Error importing langfuse.observe: {e}")

try:
    from langfuse.openai import openai

    print("✓ langfuse.openai imported successfully")
except Exception as e:
    print(f"✗ Error importing langfuse.openai: {e}")

print("\nTesting basic function with observe decorator...")


@observe()
def test_func():
    print("Inside test function")
    return "test successful"


try:
    result = test_func()
    print(f"✓ Decorated function executed successfully: {result}")
except Exception as e:
    print(f"✗ Error executing decorated function: {e}")
