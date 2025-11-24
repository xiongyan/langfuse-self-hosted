from dotenv import load_dotenv

load_dotenv()

from langfuse import observe
from langfuse.openai import openai  # OpenAI integration


@observe()
def story():
    return (
        openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is Langfuse?"}],
        )
        .choices[0]
        .message.content
    )


@observe()
def main():
    return story()


main()
