from typing import Dict, List, Optional
import os
import json
import datetime
from src.agents.triage_agent import TriageAgent
from src.agents.bias_agent import BiasAgent
from src.agents.prompt_agent import PromptAgent
from src.agents.chat_agent import ChatAgent
from src.agents.summary_agent import SummaryAgent


class Chatroom:
    """Main controller for the LLM chatroom system."""

    def __init__(self) -> None:
        self.triage_agent = TriageAgent()
        self.bias_agent = BiasAgent()
        self.prompt_agent = PromptAgent()
        self.summary_agent = SummaryAgent()
        self.chat_agents = []
        self.chat_history = []
        self.topic = ""
        self.log_file = None

    def setup_logging(self, topic: str) -> None:
        """Set up logging to a file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_topic = "".join([c if c.isalnum() else "_" for c in topic])[
            :30
        ]
        log_dir = "chat_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{log_dir}/chat_{sanitized_topic}_{timestamp}.txt"
        self.log_file = open(log_filename, "w", encoding="utf-8")
        self.log(
            f"Chat session started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.log(f"Topic: {topic}\n")

    def log(self, message: str) -> None:
        """Log a message to the file and print to console."""
        print(message)
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()  # Ensure it's written immediately

    def start_chat(self, user_input: str) -> str:
        """
        Start the chatroom process.

        Args:
            user_input: Initial input from the user

        Returns:
            The final summary
        """
        # Step 1: Triage Agent extracts topic and questions
        self.log("🔍 Triage Agent is analyzing the topic...")
        triage_output = self.triage_agent.process(user_input)
        self.topic = triage_output["topic"]
        self.setup_logging(self.topic)
        self.log(f"Topic identified: {self.topic}")
        self.log(f"Questions identified: {triage_output['questions']}\n")

        # Step 2: Bias Agent identifies perspectives
        self.log("🧠 Bias Agent is identifying perspectives...")
        bias_output = self.bias_agent.process(triage_output)
        num_perspectives = bias_output["num_perspectives"]
        self.log(f"Number of perspectives identified: {num_perspectives}")
        for i, perspective in enumerate(bias_output["perspectives"]):
            self.log(f"Perspective {i+1}: {perspective['name']}")
            self.log(f"  Description: {perspective['description']}")
            if (
                "key_arguments" in perspective
                and perspective["key_arguments"]
            ):
                self.log(
                    f"  Key Arguments: {', '.join(perspective['key_arguments'])}"
                )
            self.log("")

        # Step 3: Prompt Agent creates prompts for Chat Agents
        self.log("📝 Prompt Agent is creating system prompts...")
        prompts = self.prompt_agent.process(bias_output, triage_output)

        # Step 4: Create Chat Agents
        self.log("👥 Creating Chat Agents...")
        self.chat_agents = []
        for i, prompt_data in enumerate(prompts):
            agent_name = prompt_data.get("agent_name", f"Agent {i+1}")
            system_prompt = prompt_data.get("system_prompt", "")
            self.log(f"Created agent: {agent_name}")
            self.chat_agents.append(
                ChatAgent(name=agent_name, system_prompt=system_prompt)
            )
        self.log("")

        # Step 5: Chat Agents discuss (5 iterations)
        self.log("💬 Starting discussion...\n")

        # First round - each agent introduces their perspective
        self.log("--- Initial perspectives ---\n")
        for agent in self.chat_agents:
            response = agent.process(
                [],  # Empty chat history for first messages
                1,
                topic=self.topic,
                question=triage_output.get("questions", [""])[0],
            )
            message = {
                "agent": agent.name,
                "message": response,
                "iteration": 1,
            }
            self.chat_history.append(message)
            self.log(f"{agent.name}: {response}\n")

        # Subsequent rounds - agents respond to each other
        for iteration in range(2, 6):
            self.log(f"--- Continuing discussion (round {iteration}) ---\n")

            # Randomize the order of agents speaking for more natural flow
            import random

            speaking_order = list(self.chat_agents)
            random.shuffle(speaking_order)

            for agent in speaking_order:
                question = (
                    triage_output.get("questions", [""])[0]
                    if isinstance(triage_output.get("questions"), list)
                    else triage_output.get("questions", "")
                )
                response = agent.process(
                    self.chat_history,
                    iteration,
                    topic=self.topic,
                    question=question,
                )
                message = {
                    "agent": agent.name,
                    "message": response,
                    "iteration": iteration,
                }
                self.chat_history.append(message)
                self.log(f"{agent.name}: {response}\n")

        # Step 6: Summary Agent summarizes the discussion
        self.log("📊 Summary Agent is creating a summary...\n")
        summary = self.summary_agent.process(self.chat_history, self.topic)
        self.log(f"Summary:\n{summary}\n")
        self.log(
            f"Chat session ended at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Close log file
        if self.log_file:
            self.log_file.close()

        return summary
