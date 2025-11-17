# autogen/autogen_simple_demo.py
"""
AutoGen-style workflow for Exercise 4:
Create and refine a research paper outline for a given topic.

Backed by Groq:
- Uses GROQ_API_KEY and GROQ_MODEL from .env via config.py
"""

from typing import Dict
from pathlib import Path

from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_API_BASE,  # kept for clarity, even though Groq client doesn't need it directly
    GROQ_MODEL,
    WORKFLOW_PHASES,
    AGENT_MAX_TOKENS,
    VERBOSE,
)

# Change this topic if you like
TOPIC = "How multi-agent AI systems can support human decision-making"


class ResearchPaperWorkflow:
    def __init__(self, model: str = GROQ_MODEL):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in .env.")
        # Groq client will use its own endpoint under the hood
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = model
        self.outputs: Dict[str, str] = {}

    def run_phase(self, phase_name: str, description: str, agent_cfg: Dict):
        print("\n" + "=" * 80)
        print(f"PHASE: {description.upper()} ({phase_name})")
        print("=" * 80)

        system_prompt = (
            f"You are a {agent_cfg['role']}.\n\n"
            f"{agent_cfg.get('instructions', '')}\n\n"
            f"The paper topic is: '{TOPIC}'."
        )

        # Include previous outputs as context
        context_chunks = []
        for prev_name, content in self.outputs.items():
            context_chunks.append(f"[{prev_name.upper()} OUTPUT]\n{content}\n")
        context_text = "\n".join(context_chunks) if context_chunks else "No prior context yet."

        if phase_name == "literature":
            user_message = (
                "Write a concise literature review for this topic. "
                "Mention 3–5 key themes or directions and typical methods used."
            )
        elif phase_name == "gaps":
            user_message = (
                "Based on the literature review, identify gaps or open problems. "
                "Propose 2–3 concrete research questions or hypotheses that a new paper could address."
            )
        elif phase_name == "outline":
            user_message = (
                "Design a detailed outline for a full research paper on this topic, "
                "grounded in the research questions above. Use standard sections "
                "(Introduction, Related Work, Method, Experiments, Results/Discussion, Conclusion) "
                "with bullet points under each."
            )
        elif phase_name == "review":
            user_message = (
                "Critically review the proposed outline. Point out strengths, weaknesses, "
                "and any missing parts. Then provide an improved, final outline."
            )
        else:
            user_message = "Summarize the context above in a helpful way."

        if VERBOSE:
            print("\n[Context passed to agent]\n")
            print(context_text)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=agent_cfg.get("temperature"),
            max_tokens=AGENT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        content = response.choices[0].message.content
        self.outputs[phase_name] = content

        print("\n[Agent output]\n")
        print(content)

    def run(self):
        print("=" * 80)
        print("AUTOGEN RESEARCH PAPER OUTLINE WORKFLOW (GROQ)")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Paper topic: {TOPIC}")
        print("=" * 80)

        for phase in WORKFLOW_PHASES:
            self.run_phase(
                phase_name=phase.name,
                description=phase.description,
                agent_cfg=phase.agent_config,
            )

        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE")
        print("=" * 80)
        print("Final outline (reviewed) is in the 'review' phase output.")

        # 🔽 NEW: export all phase outputs to a file named autogen_ex4.txt
        project_root = Path(__file__).resolve().parents[1]
        out_path = project_root / "autogen_ex4.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("AutoGen Exercise 4 - Research Paper Outline\n")
            f.write("=" * 80 + "\n")
            f.write(f"Topic: {TOPIC}\n")
            f.write(f"Model: {self.model}\n\n")

            for phase_name, content in self.outputs.items():
                f.write(f"--- PHASE: {phase_name} ---\n")
                f.write(content)
                f.write("\n\n")

        print(f"\nSaved full workflow output to: {out_path}")


if __name__ == "__main__":
    workflow = ResearchPaperWorkflow()
    workflow.run()
