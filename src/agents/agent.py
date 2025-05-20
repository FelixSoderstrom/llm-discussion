from typing import Dict, List, Optional, Any
import os
import openai
import anthropic
from abc import ABC, abstractmethod


class Agent(ABC):
    """Base class for all agents in the chatroom system."""

    def __init__(self, name: str, model: str = "gpt-4o") -> None:
        """
        Initialize an agent.

        Args:
            name: The name of the agent
            model: The OpenAI model to use
        """
        self.name = name
        self.model = model
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # Initialize Anthropic client as backup
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", "")
        )
        self.use_backup = False

    def call_llm(
        self, system_prompt: str, user_message: str, temperature: float = 0.7
    ) -> str:
        """
        Call the LLM with appropriate error handling and fallback.

        Args:
            system_prompt: The system prompt to guide the LLM
            user_message: The user message to send to the LLM
            temperature: Controls randomness in the response

        Returns:
            The LLM's response as a string
        """
        try:
            if not self.use_backup:
                # Try OpenAI first
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            self.use_backup = True

        # Fallback to Anthropic if OpenAI fails
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                temperature=temperature,
                max_tokens=2000,
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            raise Exception("Both OpenAI and Anthropic APIs failed")

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and generate a response."""
        pass
