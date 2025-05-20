import os
import sys
from dotenv import load_dotenv
from src.chat.chatroom import Chatroom


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please create a .env file with your OpenAI API key.")
        sys.exit(1)

    # Welcome message
    print("\n" + "=" * 50)
    print("Welcome to the LLM Chatroom!")
    print("Enter a topic or question for the AI agents to discuss.")
    print("=" * 50 + "\n")

    # Get user input
    user_input = input("You: ")

    # Start the chatroom
    chatroom = Chatroom()
    summary = chatroom.start_chat(user_input)

    print("\n" + "=" * 50)
    print("Final Summary:")
    print(summary)
    print(
        "\nThe complete chat history has been saved to the chat_logs directory."
    )
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
