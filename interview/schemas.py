"""
CareerPilot — Mock Interview Agent Schemas (Phase 8)

Pydantic models for:
  - Interview session initialization
  - Interview question turns
  - Student answer submission
  - Turn-by-turn evaluator feedback (Technical + Communication scoring)
  - Final Comprehensive Interview Readiness Report
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class InterviewStartRequest(BaseModel):
    student_name: str = Field(..., description="Student name for memory recall and report generation")
    target_role: str = Field("Software Development Engineer (SDE-1)", description="Target job title / profile")
    technology: str = Field("Java", description="Core technology stack being interviewed on")
    total_questions: int = Field(5, ge=3, le=10, description="Total number of interview questions in the session")
    difficulty: str = Field("Medium", description="Target difficulty level (Easy, Medium, Hard, Progressive)")


class InterviewTurnQuestion(BaseModel):
    session_id: str
    question_index: int = Field(..., description="1-indexed question number")
    total_questions: int
    topic: str
    difficulty: str
    question_text: str
    interviewer_preface: Optional[str] = Field(None, description="Conversational preamble from the interviewer")
    hints: List[str] = Field(default_factory=list)


class InterviewAnswerRequest(BaseModel):
    session_id: str
    question_index: int
    student_answer: str = Field(..., min_length=2, description="Candidate's technical answer or code")


class InterviewTurnEvaluation(BaseModel):
    question_index: int
    total_questions: int
    topic: str
    technical_score: float = Field(..., ge=0.0, le=10.0, description="Technical correctness & depth (0-10)")
    communication_score: float = Field(..., ge=0.0, le=10.0, description="Clarity, structure, & articulation (0-10)")
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Combined turn score (0-10)")
    verdict: str = Field(..., description="'Strong', 'Acceptable', 'Needs Improvement'")
    feedback: str = Field(..., description="Detailed constructive feedback from the interviewer")
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    is_finished: bool = Field(False, description="True if this was the final question")
    next_question: Optional[InterviewTurnQuestion] = None


class TopicPerformance(BaseModel):
    topic: str
    average_technical_score: float
    average_communication_score: float
    verdict: str
    observations: str


class InterviewFinalSummary(BaseModel):
    session_id: str
    student_name: str
    target_role: str
    technology: str
    total_questions: int
    technical_score: float = Field(..., description="Aggregate technical percentage score (0-100)")
    communication_score: float = Field(..., description="Aggregate communication percentage score (0-100)")
    overall_interview_score: float = Field(..., description="Composite interview performance score (0-100)")
    interview_readiness_score: float = Field(..., description="Placement interview readiness index (0-100)")
    readiness_verdict: str = Field(..., description="Hiring decision: Strong Hire, Hire, Leaning Hire, No Hire")
    topic_wise_performance: List[TopicPerformance] = Field(default_factory=list)
    strong_areas: List[str] = Field(default_factory=list)
    weak_areas: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    interviewer_notes: str = Field(..., description="Senior interviewer panel summary and final assessment")
    generated_at: str
