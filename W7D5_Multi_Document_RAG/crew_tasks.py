
"""CrewAI-based analysis of retrieved RAG context."""

from crewai import Agent, Crew, LLM, Process, Task

from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL


def build_document_analysis_crew() -> Crew:
    """Create a local-LLM crew that analyzes retrieved document context."""
    local_llm = LLM(
        model=f"ollama/{OLLAMA_CHAT_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

    analyst = Agent(
        role="Document Research Analyst",
        goal=(
            "Answer questions accurately using only the supplied "
            "document excerpts and identify the supporting sources."
        ),
        backstory=(
            "You are a careful AI research analyst. You distinguish "
            "evidence from assumptions and never invent unsupported facts."
        ),
        llm=local_llm,
        verbose=False,
        allow_delegation=False,
    )

    analysis_task = Task(
        description=(
            "Answer the following question using only the supplied context.\n\n"
            "Question: {question}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Provide a concise answer and name the supporting source files. "
            "If the context is insufficient, clearly say so."
        ),
        expected_output=(
            "A concise evidence-based answer with supporting source filenames."
        ),
        agent=analyst,
    )

    return Crew(
        agents=[analyst],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=False,
    )


def analyze_with_crew(question: str, context: str) -> str:
    """Run the CrewAI analyst on a question and retrieved context."""
    if not context.strip():
        return "No retrieved context was provided for analysis."

    crew = build_document_analysis_crew()
    result = crew.kickoff(
        inputs={
            "question": question,
            "context": context,
        }
    )
    return str(result).strip()
