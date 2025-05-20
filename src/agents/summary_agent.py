from typing import Dict, List
from src.agents.agent import Agent


class SummaryAgent(Agent):
    """Agent that summarizes the chat discussion."""

    def __init__(self) -> None:
        super().__init__(name="Summary")

    def process(self, chat_history: List[Dict], original_topic: str) -> str:
        """
        Summarize the chat discussion and provide a definitive conclusion.

        Args:
            chat_history: Complete chat history
            original_topic: The original topic of discussion

        Returns:
            A summary with clear recommendations
        """
        formatted_history = "\n".join(
            [f"{msg['agent']}: {msg['message']}" for msg in chat_history]
        )

        system_prompt = """
        You are a decisive expert who analyzes discussions and provides clear, actionable conclusions.
        
        Your task is to:
        1. Briefly summarize the key points from each perspective (keep this section short)
        2. Identify areas of agreement and disagreement
        3. Most importantly, provide a DEFINITIVE RECOMMENDATION or CONCLUSION
        
        Your conclusion should be specific and actionable. If the discussion was about choosing something,
        clearly state what the best choice would be based on the discussion. Don't hedge with "it depends" -
        make a firm recommendation while acknowledging trade-offs.
        
        The recommendation should be highlighted and appear at the end of your summary.
        """

        user_message = f"""
        Original question: {original_topic}
        
        Discussion transcript:
        
        {formatted_history}
        
        Please provide a brief summary followed by a CLEAR CONCLUSION or RECOMMENDATION.
        The conclusion should directly answer the original question or resolve the topic of discussion.
        """

        return self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.3,  # Lower temperature for more consistent summarization
        )
