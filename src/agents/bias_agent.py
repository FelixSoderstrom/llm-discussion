from typing import Dict, List
from src.agents.agent import Agent


class BiasAgent(Agent):
    """Agent that identifies different perspectives/biases on a topic."""

    def __init__(self) -> None:
        super().__init__(name="Bias")

    def process(self, triage_output: Dict[str, str]) -> Dict[str, any]:
        """
        Generate different perspectives on the topic.

        Args:
            triage_output: Output from the triage agent

        Returns:
            Dict containing perspectives and the number of chat agents to create
        """
        topic = triage_output["topic"]
        questions = triage_output["questions"]

        system_prompt = """
        You are an expert at identifying different perspectives on any topic. Your task is to:
        
        1. Identify 3-5 distinct and contrasting perspectives on the given topic.
        2. Each perspective must represent a CLEARLY DIFFERENT stance or approach.
        3. For controversial topics, include perspectives from different sides of the debate.
        4. For practical questions, include multiple contrasting approaches or solutions.
        
        For each perspective:
        - Give it a short but descriptive name
        - Write a detailed description explaining this viewpoint
        - Provide 3-5 key arguments this perspective would make
        
        IMPORTANT: You MUST identify at least 3 distinct perspectives, even for seemingly simple topics.
        
        Respond with a valid JSON object with:
        {
          "perspectives": [
            {
              "name": "Perspective Name",
              "description": "Detailed description",
              "key_arguments": ["Argument 1", "Argument 2", "Argument 3"]
            },
            // Additional perspectives...
          ],
          "num_perspectives": (number between 3-5)
        }
        """

        result = self.call_llm(
            system_prompt=system_prompt,
            user_message=f"Topic: {topic}\nQuestions: {questions}",
            temperature=0.7,  # Higher temperature for more diverse perspectives
        )

        # Parse the response into a proper dict
        import json

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # Fallback if the LLM doesn't return valid JSON
            return {
                "perspectives": [
                    {
                        "name": "General Perspective",
                        "description": "Could not parse specific perspectives",
                        "key_arguments": [],
                    }
                ],
                "num_perspectives": 1,
            }
