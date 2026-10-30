"""
CareerPilot — Multi-Format Question & Readiness Report Generator (Phase 7)

Generates:
  - Placement MCQs with options, correct keys, and explanations
  - Placement Coding challenges with problem statement, starter code, and test cases
  - Conceptual deep-dive questions
  - Comprehensive Final Placement Readiness Report with company fit and advisor verdict
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger
from sqlalchemy.orm import Session

from agent.llm import get_llm
from agent.prompts import (
    CODING_QUESTION_PROMPT,
    MCQ_GENERATOR_PROMPT,
    QUESTION_GENERATOR_PROMPT,
    READINESS_REPORT_PROMPT,
)
from database.models import Answer, Evaluation, Goal, Progress, Question, Student, Topic
from database.repository import find_or_create_topic, get_or_create_student, record_question
from placement.schemas import (
    CodingDetails,
    CodingTestCase,
    MCQDetails,
    MCQOption,
    QuestionType,
    ReadinessReport,
    TopicCompetency,
    TopicPracticeResponse,
)
from rag.retriever import retrieve_context_string


def _clean_json(raw: Any) -> str:
    """Helper to safely extract JSON text from string or structured LLM response."""
    if isinstance(raw, list):
        text_parts = []
        for part in raw:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        text = "".join(text_parts).strip()
    elif isinstance(raw, str):
        text = raw.strip()
    else:
        text = str(raw).strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ── Question Generator for Specified Format ───────────────────────────────────

def generate_topic_question(
    topic: str,
    technology: str = "Java",
    question_type: QuestionType = "conceptual",
    difficulty: str = "Medium",
    context_override: Optional[str] = None,
) -> TopicPracticeResponse:
    """
    Generate an authentic placement question tailored to the chosen format:
    MCQ, Coding challenge, or Conceptual interview question.
    """
    clean_topic = topic.strip() or "Core Fundamentals"
    logger.info(f"Generating {question_type.upper()} question for '{clean_topic}' ({technology})")

    # 1. Retrieve RAG context if not provided
    study_context = context_override
    if study_context is None:
        try:
            study_context = retrieve_context_string(f"{clean_topic} {technology}", top_k=3)
        except Exception as exc:
            logger.warning(f"RAG retrieval skipped for topic question: {exc}")
            study_context = "No specific study document notes found."



    # 2. MCQ Question Generation
    if question_type == "mcq":
        prompt_val = MCQ_GENERATOR_PROMPT.format_messages(
            technology=technology,
            topic=clean_topic,
            difficulty=difficulty,
            study_material=study_context or "General core language knowledge.",
        )
        try:
            llm = get_llm()
            res = llm.invoke(prompt_val)
            parsed = json.loads(_clean_json(res.content))

            raw_options = parsed.get("options", [])
            options = [
                MCQOption(label=opt.get("label", chr(65 + i)), text=opt.get("text", ""))
                for i, opt in enumerate(raw_options)
            ]
            mcq_details = MCQDetails(
                options=options,
                correct_option=parsed.get("correct_option", "A").upper(),
                explanation=parsed.get("explanation", "Standard placement core concept."),
            )
            return TopicPracticeResponse(
                success=True,
                topic=clean_topic,
                question_type="mcq",
                difficulty=difficulty,
                question_text=parsed.get("question_text", f"Select the correct statement regarding {clean_topic}."),
                mcq_details=mcq_details,
                hints=parsed.get("hints", []),
                ideal_answer=mcq_details.explanation,
                retrieved_context=study_context,
            )
        except Exception as exc:
            logger.warning(f"Failed to generate dynamic MCQ, using structured fallback: {exc}")
            # Reliable fallback
            fallback_options = [
                MCQOption(label="A", text=f"{clean_topic} enables compile-time type safety and code reusability."),
                MCQOption(label="B", text=f"{clean_topic} is completely resolved at runtime without compiler checks."),
                MCQOption(label="C", text=f"{clean_topic} disables automated memory management in {technology}."),
                MCQOption(label="D", text=f"{clean_topic} can only be applied to static primitive variables."),
            ]
            return TopicPracticeResponse(
                success=True,
                topic=clean_topic,
                question_type="mcq",
                difficulty=difficulty,
                question_text=f"Which of the following statements best describes the primary advantage of {clean_topic} in {technology}?",
                mcq_details=MCQDetails(
                    options=fallback_options,
                    correct_option="A",
                    explanation=f"{clean_topic} ensures robust type safety and predictable modular architecture.",
                ),
                hints=[f"Consider how {technology} manages type checking at compile time."],
                ideal_answer="A",
                retrieved_context=study_context,
            )

    # 3. Coding Challenge Generation
    elif question_type == "coding":
        prompt_val = CODING_QUESTION_PROMPT.format_messages(
            technology=technology,
            topic=clean_topic,
            difficulty=difficulty,
            study_material=study_context or "General algorithmic concepts.",
        )
        try:
            llm = get_llm()
            res = llm.invoke(prompt_val)
            parsed = json.loads(_clean_json(res.content))

            raw_cases = parsed.get("sample_test_cases", [])
            test_cases = [
                CodingTestCase(
                    input=str(tc.get("input", "")),
                    expected_output=str(tc.get("expected_output", "")),
                    explanation=tc.get("explanation"),
                )
                for tc in raw_cases
            ]
            coding_details = CodingDetails(
                problem_statement=parsed.get("problem_statement", parsed.get("question_text", "")),
                constraints=parsed.get("constraints", ["Time Limit: 1.0s", "Memory Limit: 256 MB"]),
                starter_code=parsed.get("starter_code", f"// Solve {clean_topic} in {technology}\nclass Solution {{\n    // your code\n}}"),
                sample_test_cases=test_cases,
                optimal_time_complexity=parsed.get("optimal_time_complexity", "O(N)"),
                optimal_space_complexity=parsed.get("optimal_space_complexity", "O(1)"),
            )
            return TopicPracticeResponse(
                success=True,
                topic=clean_topic,
                question_type="coding",
                difficulty=difficulty,
                question_text=parsed.get("question_text", f"Implement optimal solution for {clean_topic}"),
                coding_details=coding_details,
                hints=parsed.get("hints", []),
                ideal_answer=parsed.get("ideal_answer"),
                retrieved_context=study_context,
            )
        except Exception as exc:
            logger.warning(f"Failed to generate dynamic coding challenge, using structured fallback: {exc}")
            starter = f"class Solution {{\n    public boolean solve({technology} input) {{\n        // Write solution for {clean_topic}\n        return true;\n    }}\n}}"
            test_cases = [
                CodingTestCase(input="Input Case 1", expected_output="true", explanation="Satisfies criteria."),
                CodingTestCase(input="Empty or null input", expected_output="false", explanation="Edge case handling."),
            ]
            return TopicPracticeResponse(
                success=True,
                topic=clean_topic,
                question_type="coding",
                difficulty=difficulty,
                question_text=f"Implement an optimal algorithmic routine demonstrating {clean_topic} in {technology}.",
                coding_details=CodingDetails(
                    problem_statement=f"Given an input scenario requiring {clean_topic}, write an optimal function that returns the desired placement outcome.",
                    constraints=["1 <= N <= 10^5", "All values are within 32-bit signed integers"],
                    starter_code=starter,
                    sample_test_cases=test_cases,
                    optimal_time_complexity="O(N)",
                    optimal_space_complexity="O(1)",
                ),
                hints=["Check for edge cases such as empty input or boundary values."],
                ideal_answer=starter,
                retrieved_context=study_context,
            )

    # 4. Conceptual Interview Question Generation (Default)
    else:
        prompt_val = QUESTION_GENERATOR_PROMPT.format_messages(
            technology=technology,
            topic=clean_topic,
            difficulty=difficulty,
            is_remedial=False,
            study_material=study_context or "General placement concepts.",
        )
        try:
            llm = get_llm()
            res = llm.invoke(prompt_val)
            parsed = json.loads(_clean_json(res.content))
            return TopicPracticeResponse(
                success=True,
                topic=clean_topic,
                question_type="conceptual",
                difficulty=difficulty,
                question_text=parsed.get("question_text", f"Explain the internal mechanics of {clean_topic}."),
                hints=[parsed["hint"]] if parsed.get("hint") else [],
                ideal_answer=parsed.get("sample_ideal_answer"),
                retrieved_context=study_context,
            )
        except Exception as exc:
            logger.warning(f"Failed to generate dynamic conceptual question, using structured fallback: {exc}")
            return TopicPracticeResponse(
                success=True,
                topic=clean_topic,
                question_type="conceptual",
                difficulty=difficulty,
                question_text=f"Explain how {clean_topic} operates under the hood in {technology}. What are its primary trade-offs and memory implications?",
                hints=[f"Consider how the runtime engine/compiler implements {clean_topic}."],
                ideal_answer=f"{clean_topic} provides modular design, predictable execution, and optimized runtime performance.",
                retrieved_context=study_context,
            )


# ── Final Placement Readiness Report Generator ────────────────────────────────

def generate_readiness_report(
    db: Session,
    student_name: str,
    target_technology: str = "Java",
    target_role: str = "Software Engineer",
) -> ReadinessReport:
    """
    Compile complete historical evaluations, mastery metrics, and generate an
    executive-level Placement Readiness Report for the student.
    """
    clean_name = student_name.strip() or "Student"
    student = db.query(Student).filter(Student.name.ilike(clean_name)).first()

    if not student:
        student = get_or_create_student(db, clean_name, target_technology)

    # Fetch progress and history
    prog = db.query(Progress).filter(Progress.student_id == student.id).first()
    answers = (
        db.query(Answer)
        .filter(Answer.student_id == student.id)
        .all()
    )

    mastered = prog.mastered_topics if prog else []
    weak = prog.weak_topics if prog else []
    topic_scores = prog.topic_scores if prog else {}
    total_q = len(answers)

    # Compute competencies for each topic attempted or in progress
    competencies: list[TopicCompetency] = []
    all_known_topics = set(mastered + weak + list(topic_scores.keys()))
    if not all_known_topics:
        all_known_topics = {"Core Language Fundamentals", "Object-Oriented Design"}

    for t in sorted(all_known_topics):
        scores_for_t = topic_scores.get(t, [])
        avg = round(sum(scores_for_t) / len(scores_for_t), 1) if scores_for_t else (8.0 if t in mastered else 5.0)

        if t in mastered or avg >= 7.0:
            status_val = "Mastered"
            rec = "Maintained high technical depth. Ready for technical interview rounds."
        elif t in weak or avg < 6.0:
            status_val = "Weak"
            rec = "Focus on foundational mechanisms and practice at least 5 medium interview problems."
        else:
            status_val = "Needs Practice"
            rec = "Good conceptual grasp; review edge cases and time complexity."

        competencies.append(
            TopicCompetency(
                topic=t,
                status=status_val,
                attempts_count=len(scores_for_t) or 1,
                average_score=avg,
                recommendation=rec,
            )
        )

    # Deterministic base readiness score
    mastered_count = len(mastered)
    weak_count = len(weak)
    overall_avg = prog.average_score if prog else 7.0

    # Score calculation: 50% mastery breadth, 40% performance score, 10% volume
    breadth_score = min(50.0, (mastered_count / max(1, (mastered_count + weak_count))) * 50.0) if (mastered_count + weak_count) > 0 else 25.0
    perf_score = min(40.0, (overall_avg / 10.0) * 40.0)
    vol_score = min(10.0, (total_q / 5.0) * 10.0)
    calc_score = int(breadth_score + perf_score + vol_score)
    calc_score = max(20, min(95, calc_score))


    prompt_val = READINESS_REPORT_PROMPT.format_messages(
        student_name=clean_name,
        target_role=target_role,
        technology=target_technology,
        mastered_topics=", ".join(mastered) or "None recorded",
        weak_topics=", ".join(weak) or "None currently flagged",
        topic_scores=json.dumps(topic_scores),
        total_questions=total_q,
        average_score=overall_avg,
    )

    try:
        res = llm.invoke(prompt_val)
        parsed = json.loads(_clean_json(res.content))

        return ReadinessReport(
            student_name=clean_name,
            target_technology=target_technology,
            target_role=target_role,
            overall_readiness_score=parsed.get("overall_readiness_score", calc_score),
            readiness_level=parsed.get("readiness_level", "Placement Ready" if calc_score >= 75 else "Developing"),
            company_fit=parsed.get("company_fit", ["Tier-1 Product Companies", "High-growth Startups", "IT Services MNCs"]),
            total_questions_attempted=total_q,
            total_topics_mastered=len(mastered),
            total_weak_topics=len(weak),
            competencies=competencies,
            demonstrated_strengths=parsed.get("demonstrated_strengths", [
                "Strong conceptual grasp of core architecture",
                "Structured approach to answering technical questions",
            ]),
            critical_growth_areas=parsed.get("critical_growth_areas", (
                [f"Deepen edge-case handling in {w}" for w in weak] if weak else ["Practice timed coding challenges"]
            )),
            placement_advisor_verdict=parsed.get("placement_advisor_verdict", (
                f"{clean_name} has built a solid foundation in {target_technology} with {mastered_count} mastered topics. "
                "With continued practice on edge cases and coding speed, the candidate is well-positioned for placement offers."
            )),
            action_plan_next_steps=parsed.get("action_plan_next_steps", [
                "Review weak topics identified in your diagnostic record",
                "Practice 2-3 timed coding challenges under interview conditions",
                "Conduct a mock interview session before your real placement round",
            ]),
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
    except Exception as exc:
        logger.warning(f"Failed to generate dynamic readiness report, using deterministic analytics: {exc}")
        level = "Placement Ready" if calc_score >= 75 else ("Strong Candidate" if calc_score >= 60 else "Developing")
        return ReadinessReport(
            student_name=clean_name,
            target_technology=target_technology,
            target_role=target_role,
            overall_readiness_score=calc_score,
            readiness_level=level,
            company_fit=["Tier-1 Product Companies", "High-Growth Tech Startups", "IT Service MNCs"],
            total_questions_attempted=total_q,
            total_topics_mastered=len(mastered),
            total_weak_topics=len(weak),
            competencies=competencies,
            demonstrated_strengths=[
                f"Consistent technical preparation in {target_technology}",
                f"Successfully demonstrated competence across {len(mastered)} key curriculum areas",
            ],
            critical_growth_areas=[f"Address weak areas in {w}" for w in weak] if weak else ["Refine time complexity communication"],
            placement_advisor_verdict=(
                f"{clean_name} shows steady progress toward {target_role} readiness with a calculated readiness index of {calc_score}%. "
                "The candidate should maintain daily practice on diagnostic weaknesses to secure top offers."
            ),
            action_plan_next_steps=[
                "Complete targeted remediation on all remaining weak topics",
                "Practice whiteboard coding and articulate trade-offs aloud",
                "Simulate full mock interview rounds in CareerPilot",
            ],
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
