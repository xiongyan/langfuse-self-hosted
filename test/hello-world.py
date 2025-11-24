from langfuse import observe, get_client


@observe
def my_function():
    return "Hello, world!"  # Input/output and timings are automatically captured


my_function()

# Flush events in short-lived applications
langfuse = get_client()
langfuse.flush()
