from typing import Dict, List
from src.agents.agent import Agent


class PromptAgent(Agent):
    """Agent that creates system prompts for chat agents based on biases."""

    def __init__(self) -> None:
        super().__init__(name="Prompt")

    def process(self, bias_output: Dict, triage_output: Dict) -> List[Dict]:
        """
        Create system prompts for each perspective.

        Args:
            bias_output: Output from the bias agent
            triage_output: Output from the triage agent

        Returns:
            List of system prompts for chat agents
        """
        topic = triage_output["topic"]
        questions = triage_output["questions"]
        perspectives = bias_output["perspectives"]

        system_prompt = """
        You are creating system prompts for AI agents who will participate in a online forum discussion.
        
        CRITICAL REQUIREMENTS:
        1. Create prompts that will make agents interact like REAL PEOPLE in a forum thread
        2. Prompts should enable a distinct writing style and personality traits
        3. The prompt needs to limit the number of words to 70.
        4. The prompt should include instructions to interact with other agents and respond directly to them.
        5. Include instructions for casual language, slang, abbreviations.
        6. Each agent should occasionally DISAGREE with others and DEFEND their perspective
        7. Agents should have authentic human quirks (impatience, humor, skepticism, etc.)
        
        For each perspective, create a JSON object with:
        - 'agent_name': A realistic username that reflects their perspective
        - 'system_prompt': Instructions for HOW they should write and interact
        """

        # Prepare the perspectives info for the prompt
        perspectives_info = "\n\n".join(
            [
                f"Perspective {i+1}: {p['name']}\nDescription: {p['description']}\nKey Arguments: {', '.join(p['key_arguments'] if isinstance(p.get('key_arguments'), list) else [])}"
                for i, p in enumerate(perspectives)
            ]
        )

        result = self.call_llm(
            system_prompt=system_prompt,
            user_message=f"Topic: {topic}\nQuestions: {questions}\n\nPerspectives:\n{perspectives_info}",
            temperature=0.5,
        )

        # Parse the response
        import json

        try:
            prompts = json.loads(result)
            if isinstance(prompts, list):
                return prompts
            elif isinstance(prompts, dict) and "prompts" in prompts:
                return prompts["prompts"]
            else:
                raise ValueError("Unexpected format in prompt agent response")
        except (json.JSONDecodeError, ValueError):
            # Fallback
            return [
                {
                    "agent_name": p["name"],
                    "system_prompt": f"You are an expert representing the {p['name']} perspective on {topic}. Your view is: {p['description']}. Discuss this topic thoughtfully while staying true to your perspective.",
                }
                for p in perspectives
            ]
