"""
CrewAI Multi-Agent Demo: 3-Day Conference Agenda Planner (Groq-backed)
=====================================================================

Exercise 4 scenario:
Plan a 3-day conference agenda for a given theme using multiple agents.

Agents:
1. ProgramChairAgent    - defines goals, audience, and tracks
2. TrackDesignerAgent   - proposes sessions & talks per track
3. SchedulePlannerAgent - builds the day-by-day schedule
4. LogisticsAgent       - checks for conflicts & logistical issues
"""

from datetime import datetime
from pathlib import Path
import os
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "autogen"))

from crewai import Agent, Task, Crew

from config import GROQ_API_KEY, GROQ_API_BASE, GROQ_MODEL, VERBOSE


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_program_chair_agent(conference_theme: str) -> Agent:
    return Agent(
        role="Program Chair",
        goal=(
            f"Define clear goals, target audience, and high-level tracks for a "
            f"3-day conference on '{conference_theme}'."
        ),
        backstory=(
            "You have chaired multiple academic and industry conferences. "
            "You are skilled at clarifying who the conference is for, what "
            "its objectives are, and how to structure tracks that align with "
            "those goals."
        ),
        verbose=VERBOSE,
        allow_delegation=False,
    )


def create_track_designer_agent(conference_theme: str) -> Agent:
    return Agent(
        role="Track & Session Designer",
        goal=(
            f"Design coherent session themes and talk ideas that fit the tracks "
            f"for a 3-day '{conference_theme}' conference."
        ),
        backstory=(
            "You specialize in turning broad tracks into concrete sessions. "
            "You ensure each session has a clear focus, a mix of perspectives, "
            "and a logical flow for attendees."
        ),
        verbose=VERBOSE,
        allow_delegation=False,
    )


def create_schedule_planner_agent() -> Agent:
    return Agent(
        role="Schedule Planner",
        goal=(
            "Arrange sessions into a practical 3-day timetable, balancing keynotes, "
            "talks, breaks, and networking slots."
        ),
        backstory=(
            "You are experienced in agenda planning. You avoid overloading any day, "
            "make sure breaks are reasonable, and space popular sessions to avoid conflicts."
        ),
        verbose=VERBOSE,
        allow_delegation=False,
    )


def create_logistics_agent() -> Agent:
    return Agent(
        role="Logistics & Risk Reviewer",
        goal=(
            "Review the proposed agenda for conflicts and logistical issues, and "
            "suggest improvements."
        ),
        backstory=(
            "You think like an operations manager. You spot problems such as "
            "double-booked rooms, back-to-back intense sessions, or accessibility "
            "issues, and propose practical fixes."
        ),
        verbose=VERBOSE,
        allow_delegation=False,
    )


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_program_task(program_chair: Agent, conference_theme: str) -> Task:
    return Task(
        description=(
            f"Define the high-level structure of a 3-day conference on '{conference_theme}'. "
            "Clarify the primary audience, main goals, and propose 3–4 tracks "
            "(e.g., technical, applications, ethics, workshops)."
        ),
        agent=program_chair,
        expected_output=(
            "A short document with: (1) conference goals, (2) target audience, "
            "(3) 3–4 named tracks with 1–2 sentences describing each."
        ),
    )


def create_track_task(track_designer: Agent) -> Task:
    return Task(
        description=(
            "Based on the defined tracks and goals, design 6–8 sessions per day "
            "for a 3-day conference. For each session, provide a title, 2–3 bullet "
            "points on content, and which track it belongs to."
        ),
        agent=track_designer,
        expected_output=(
            "A list of proposed sessions grouped by track, with titles and brief descriptions."
        ),
    )


def create_schedule_task(schedule_planner: Agent) -> Task:
    return Task(
        description=(
            "Convert the proposed sessions into a concrete 3-day agenda. "
            "Assume each day runs roughly 9:00–17:30 with a keynote, "
            "two morning sessions, lunch, two afternoon sessions, and an optional "
            "evening event. Assign sessions to time slots and mention where parallel "
            "tracks exist."
        ),
        agent=schedule_planner,
        expected_output=(
            "A day-by-day agenda (Day 1, Day 2, Day 3) with time slots, session titles, "
            "and indications of parallel tracks when applicable."
        ),
    )


def create_logistics_task(logistics_agent: Agent) -> Task:
    return Task(
        description=(
            "Review the 3-day agenda for issues such as too many parallel sessions, "
            "insufficient breaks, or clustering similar content. Suggest concrete "
            "improvements and provide a final, adjusted agenda."
        ),
        agent=logistics_agent,
        expected_output=(
            "A short critique of the agenda highlighting problems, followed by "
            "an improved agenda layout."
        ),
    )


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main(conference_theme: str = "AI in Education"):
    print("=" * 80)
    print("CrewAI Multi-Agent Conference Agenda Planner (Groq)")
    print("=" * 80)
    print(f"Conference Theme: {conference_theme}")
    print("Duration: 3 days\n")

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in .env.")

    # Bind Groq API into the OpenAI-compatible env vars that CrewAI expects
    os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
    os.environ["OPENAI_API_BASE"] = GROQ_API_BASE
    os.environ["OPENAI_MODEL_NAME"] = GROQ_MODEL

    # Create agents
    program_chair = create_program_chair_agent(conference_theme)
    track_designer = create_track_designer_agent(conference_theme)
    schedule_planner = create_schedule_planner_agent()
    logistics_agent = create_logistics_agent()

    # Create tasks
    program_task = create_program_task(program_chair, conference_theme)
    track_task = create_track_task(track_designer)
    schedule_task = create_schedule_task(schedule_planner)
    logistics_task = create_logistics_task(logistics_agent)

    print("Forming the Conference Planning Crew...\n")
    crew = Crew(
        agents=[program_chair, track_designer, schedule_planner, logistics_agent],
        tasks=[program_task, track_task, schedule_task, logistics_task],
        verbose=VERBOSE,
        process="sequential",
    )

    print("Starting Crew execution...\n")
    result = crew.kickoff(inputs={"conference_theme": conference_theme})

    print("\n" + "=" * 80)
    print("FINAL 3-DAY CONFERENCE AGENDA (SUMMARY)")
    print("=" * 80)
    print(result)

    # 🔽 NEW: write summary to crewai_demo_ex4.txt in project root
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "crewai_demo_ex4.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("CrewAI Exercise 4 - 3-Day Conference Agenda\n")
        f.write("=" * 80 + "\n")
        f.write(f"Conference Theme: {conference_theme}\n")
        f.write(f"Model (Groq): {GROQ_MODEL}\n")
        f.write(f"Generated at: {datetime.now()}\n\n")
        f.write(str(result))

    print(f"\nSaved conference agenda summary to: {out_path}")


if __name__ == "__main__":
    import sys

    theme = "AI in Education"
    if len(sys.argv) > 1:
        theme = " ".join(sys.argv[1:])
    main(theme)
