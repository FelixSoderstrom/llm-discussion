from typing import Dict
from src.agents.agent import Agent
import re


class TriageAgent(Agent):
    """Agent that identifies the main topic and questions from user input."""

    def __init__(self) -> None:
        super().__init__(name="Triage")

    def process(self, user_input: str) -> Dict[str, str]:
        """
        Extract the main topic and questions from user input.

        Args:
            user_input: The raw input from the user

        Returns:
            Dict containing the extracted topic and questions
        """
        system_prompt = """
        You are a triage agent that carefully analyzes user input to extract:
        1. The main topic of discussion
        2. The specific questions or problems that need answering
        
        Respond ONLY with a valid JSON object with the following structure:
        {
          "topic": "Clear topic name, summarized in 3-5 words",
          "questions": ["Question 1", "Question 2", ...] or a single string for one question
        }
        
        If there's only one question, put it in an array with one element.
        Make sure your response is valid JSON that can be parsed by Python's json.loads().
        Do not include any explanation or additional text outside the JSON object.
        """

        result = self.call_llm(
            system_prompt=system_prompt,
            user_message=f"Extract the topic and questions from this input:\n\n{user_input}",
            temperature=0.3,
        )

        # Handle JSON parsing more robustly
        import json
        import re

        # Attempt to find JSON in the response
        json_match = re.search(
            r"(\{.*\})", result.replace("\n", " "), re.DOTALL
        )

        try:
            if json_match:
                parsed_result = json.loads(json_match.group(1))
            else:
                parsed_result = json.loads(result)

            # Ensure questions is always a list
            if isinstance(parsed_result.get("questions"), str):
                parsed_result["questions"] = [parsed_result["questions"]]

            return parsed_result
        except json.JSONDecodeError:
            # Fallback with manual extraction
            return {
                "topic": self._extract_topic(result) or "Undefined topic",
                "questions": [
                    self._extract_question(result)
                    or "No specific question identified"
                ],
            }

    def _extract_topic(self, text: str) -> str:
        """Extract topic from text when JSON parsing fails."""
        topic_match = re.search(
            r'(?:topic|Topic):\s*["\']*([^"\'\n]+)["\'\n]', text
        )
        if topic_match:
            return topic_match.group(1).strip()
        return None

    def _extract_question(self, text: str) -> str:
        """Extract question from text when JSON parsing fails."""
        question_match = re.search(
            r'(?:question|Question)[s]*:\s*["\']*([^"\'\n]+)["\'\n]', text
        )
        if question_match:
            return question_match.group(1).strip()
        return None
