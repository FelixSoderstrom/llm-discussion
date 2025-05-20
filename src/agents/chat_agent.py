from typing import Dict, List, Optional
from src.agents.agent import Agent


class ChatAgent(Agent):
    """Agent that participates in the chatroom discussion."""

    def __init__(
        self, name: str, system_prompt: str, model: str = "gpt-4o"
    ) -> None:
        super().__init__(name=name, model=model)
        self.system_prompt = system_prompt
        self.agent_id = name.lower().replace(" ", "_")

    def process(
        self,
        chat_history: List[Dict],
        iteration: int,
        topic: str = "",
        question: str = "",
    ) -> str:
        """
        Generate a response based on the chat history.

        Args:
            chat_history: List of previous messages in the chat
            iteration: Current iteration number
            topic: The main topic of discussion
            question: The specific question being discussed

        Returns:
            The agent's response
        """
        formatted_history = "\n".join(
            [f"{msg['agent']}: {msg['message']}" for msg in chat_history]
        )

        # Determine if this is the first message or a response
        if len(chat_history) == 0:
            user_message = f"""
            TOPIC: {topic}
            QUESTION: {question}
            
            You are the first speaker in this discussion. 
            
            CRITICAL: Your response MUST be under 50 words MAXIMUM. This is non-negotiable.
            Write like a real person in a forum - casual, brief, and to the point.
            """
        else:
            # Find messages to potentially respond to
            other_messages = [
                msg for msg in chat_history if msg["agent"] != self.name
            ]
            response_target = ""

            if other_messages:
                # Randomly decide whether to respond to the most recent message or an earlier one
                import random

                if len(other_messages) > 1 and random.random() < 0.4:
                    # Sometimes respond to an earlier message for more natural conversation flow
                    target_msg = random.choice(other_messages[:-1])
                else:
                    # Usually respond to the most recent message
                    target_msg = other_messages[-1]

                response_target = f"@{target_msg['agent']}"

            user_message = f"""
            TOPIC: {topic}
            QUESTION: {question}
            
            Current conversation:
            
            {formatted_history}
            
            CRITICAL INSTRUCTIONS:
            1. Start your response with "{response_target}" to directly address another user
            2. Your ENTIRE response MUST be UNDER 50 WORDS - no exceptions!
            3. Be casual and conversational like a real forum post
            4. Have a strong opinion - agree or disagree with something specific
            5. Do not introduce yourself or be overly formal
            
            Write your short forum reply now:
            """

        response = self.call_llm(
            system_prompt=self.system_prompt,
            user_message=user_message,
            temperature=0.9,  # Higher temperature for more diverse responses
        )

        # Programmatically enforce word limit
        words = response.split()
        if len(words) > 70:
            return " ".join(words[:65]) + "..."

        return response
