"""
CareerPilot — Placement Features Package (Phase 7)

Provides:
  - Multi-format practice question schemas (MCQ, Coding, Conceptual)
  - Topic-wise preparation selector
  - Final Placement Readiness Report analyzer & generator
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


QuestionType = Literal["mcq", "coding", "conceptual"]


class MCQOption(BaseModel):
    label: str = Field(..., description="Option letter: A, B, C, or D")
    text: str = Field(..., description="Option text")


class MCQDetails(BaseModel):
    options: list[MCQOption] = Field(..., description="List of 4 options A, B, C, D")
    correct_option: str = Field(..., description="Correct option label: A, B, C, or D")
    explanation: str = Field(..., description="Explanation of why this option is correct")


class CodingTestCase(BaseModel):
    input: str
    expected_output: str
    explanation: Optional[str] = None


class CodingDetails(BaseModel):
    problem_statement: str
    constraints: list[str] = []
    starter_code: Optional[str] = None
    sample_test_cases: list[CodingTestCase] = []
    optimal_time_complexity: Optional[str] = None
    optimal_space_complexity: Optional[str] = None


class TopicPracticeRequest(BaseModel):
    student_name: str = "Student"
    technology: str = "Java"
    topic: str
    question_type: QuestionType = "conceptual"
    difficulty: str = "Medium"
    session_id: Optional[str] = None


class TopicPracticeResponse(BaseModel):
    success: bool
    topic: str
    question_type: QuestionType
    difficulty: str
    question_text: str
    mcq_details: Optional[MCQDetails] = None
    coding_details: Optional[CodingDetails] = None
    hints: list[str] = []
    ideal_answer: Optional[str] = None
    retrieved_context: Optional[str] = None
    error: Optional[str] = None


class ReadinessReportRequest(BaseModel):
    student_name: str
    target_technology: Optional[str] = None
    target_role: Optional[str] = "Software Engineer"


class TopicCompetency(BaseModel):
    topic: str
    status: Literal["Mastered", "Needs Practice", "Weak"]
    attempts_count: int
    average_score: float
    recommendation: str


class ReadinessReport(BaseModel):
    student_name: str
    target_technology: str
    target_role: str
    overall_readiness_score: int = Field(..., ge=0, le=100, description="Readiness score from 0 to 100%")
    readiness_level: str = Field(..., description="e.g. Placement Ready, Interview Ready, Needs Improvement")
    company_fit: list[str] = Field(..., description="Target tiers e.g. Tier 1 Product, High-growth Tech, Service MNCs")
    total_questions_attempted: int
    total_topics_mastered: int
    total_weak_topics: int
    competencies: list[TopicCompetency] = []
    demonstrated_strengths: list[str] = []
    critical_growth_areas: list[str] = []
    placement_advisor_verdict: str
    action_plan_next_steps: list[str] = []
    generated_at: str
