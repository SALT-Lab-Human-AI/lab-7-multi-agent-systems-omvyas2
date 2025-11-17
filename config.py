# config.py
"""
Shared configuration for AutoGen and CrewAI demos (Groq-only).

Reads from .env:
- GROQ_API_KEY
- GROQ_MODEL (optional, defaults to llama-3.1-8b-instant)
- AGENT_TEMPERATURE
- AGENT_MAX_TOKENS
- AGENT_TIMEOUT
- VERBOSE, DEBUG
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
env_path = ROOT_DIR / ".env"
load_dotenv(env_path)

# ---------------------------------------------------------------------
# Groq config
# ---------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Allow override via env; otherwise use a reasonable default Groq chat model
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Groq's OpenAI-compatible base URL
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# ---------------------------------------------------------------------
# Agent-level config
# ---------------------------------------------------------------------
AGENT_TEMPERATURE_DEFAULT = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "2000"))
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300"))

VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ---------------------------------------------------------------------
# AutoGen-style workflow config (Exercise 4: research paper outline)
# ---------------------------------------------------------------------

LITERATURE_AGENT = {
    "name": "LiteratureReviewer",
    "role": "Literature Review Specialist",
    "temperature": AGENT_TEMPERATURE_DEFAULT,
    "instructions": (
        "You conduct a concise but insightful literature review. "
        "You summarize key themes, representative papers, and open questions, "
        "highlighting where the field currently stands."
    ),
}

GAP_AGENT = {
    "name": "GapAnalyst",
    "role": "Research Gap Analyst",
    "temperature": AGENT_TEMPERATURE_DEFAULT,
    "instructions": (
        "You read the literature summary and identify gaps, tensions, and "
        "promising research directions. You propose 2–3 concrete research "
        "questions or hypotheses that a new paper could address."
    ),
}

OUTLINE_AGENT = {
    "name": "OutlineDesigner",
    "role": "Research Paper Outline Designer",
    "temperature": AGENT_TEMPERATURE_DEFAULT,
    "instructions": (
        "You turn a research idea into a well-structured paper outline. "
        "You define sections (e.g., Introduction, Related Work, Method, "
        "Experiments, Discussion, Conclusion) and list bullet points for "
        "what each section should cover."
    ),
}

REVIEW_AGENT = {
    "name": "OutlineReviewer",
    "role": "Critical Research Mentor",
    "temperature": AGENT_TEMPERATURE_DEFAULT,
    "instructions": (
        "You critically review the proposed outline, checking for coherence, "
        "feasibility, and novelty. You suggest improvements and highlight "
        "any missing sections or clarifications needed."
    ),
}


@dataclass
class WorkflowPhase:
    name: str
    description: str
    agent_config: Dict


WORKFLOW_PHASES: List[WorkflowPhase] = [
    WorkflowPhase(
        name="literature",
        description="Literature review on the topic",
        agent_config=LITERATURE_AGENT,
    ),
    WorkflowPhase(
        name="gaps",
        description="Analyze research gaps and propose questions",
        agent_config=GAP_AGENT,
    ),
    WorkflowPhase(
        name="outline",
        description="Design structured paper outline",
        agent_config=OUTLINE_AGENT,
    ),
    WorkflowPhase(
        name="review",
        description="Critically review and refine the outline",
        agent_config=REVIEW_AGENT,
    ),
]
