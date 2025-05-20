# LLM Chatroom

A command-line application that creates a simulated discussion between AI agents on a topic or question provided by the user.

## Overview

LLM Chatroom uses OpenAI's API to simulate a conversation between multiple AI agents. The agents discuss the provided topic, offer different perspectives, and generate a final summary of the conversation. This can be useful for:

- Exploring different viewpoints on a complex topic
- Generating ideas through simulated brainstorming
- Understanding nuanced arguments around a subject
- Teaching through simulated discussions

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/FelixSoderstrom/llm-discussion.git
   cd llm-discussion
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the application from the command line:

```
python main.py
```

When prompted, enter a topic or question for the AI agents to discuss. For example:
- "What are the ethical implications of AI development?"
- "Discuss the pros and cons of remote work"
- "How might climate change affect global agriculture in the next 50 years?"

The AI agents will conduct a discussion, and the application will:
1. Display the conversation in real-time
2. Generate a final summary of the discussion
3. Save the complete chat history to the `chat_logs` directory

## Project Structure

```
llm-discussion/
├── .env                  # Environment variables including API keys
├── main.py               # Entry point for the application
├── requirements.txt      # Python dependencies
├── chat_logs/            # Directory containing saved chat histories
└── src/
    └── chat/
        ├── chatroom.py   # Chatroom class that manages the AI discussion
        └── ...           # Other modules related to the chat functionality
```

## Requirements

- Python 3.6+
- OpenAI API key
- Required Python packages (see requirements.txt)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 