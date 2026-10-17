"""
CareerPilot — Sample Test Data Fixtures (Phase 9 — Testing)

Provides curated datasets for:
  - Diverse student placement preparation profiles
  - Multidimensional answers (High-scoring, Borderline, Weak, and Edge Cases)
  - Sample test documents for ingestion & semantic retrieval
  - Sample curriculum topics and multi-format questions
"""

from __future__ import annotations

from typing import Any, Dict, List

# ── Sample Student Profiles ───────────────────────────────────────────────────

SAMPLE_STUDENTS: List[Dict[str, Any]] = [
    {
        "name": "Ananya Sharma",
        "email": "ananya.sharma@example.edu",
        "target_technology": "Java",
        "available_days": 7,
        "goal": "Prepare me for a Tier-1 Product Company Java backend interview in 7 days",
    },
    {
        "name": "Vikramaditya Roy",
        "email": "vikram.roy@example.edu",
        "target_technology": "Python",
        "available_days": 3,
        "goal": "I have an Infosys Specialist Programmer interview in 3 days. Focus on Data Structures and Algorithms.",
    },
    {
        "name": "Neha Kulkarni",
        "email": "neha.k@example.edu",
        "target_technology": "Java",
        "available_days": 14,
        "goal": "Master Java Concurrency, Collections framework, and JVM Internals for campus placements in 2 weeks.",
    },
]

# ── Sample Technical Answers for Evaluation Testing ───────────────────────────

SAMPLE_ANSWERS: Dict[str, Dict[str, Any]] = {
    "strong_oop": {
        "topic": "Object-Oriented Programming",
        "question": "Explain dynamic method dispatch and how runtime polymorphism is achieved in Java.",
        "answer": (
            "Dynamic method dispatch is the mechanism by which a call to an overridden method is resolved "
            "at runtime rather than compile time. In Java, when an overridden method is called through a superclass reference, "
            "the JVM determines which version of that method to execute based on the actual object type being referred to, "
            "using the vtable (virtual method table) in the method area. This satisfies the Liskov Substitution Principle."
        ),
        "expected_min_score": 8.0,
        "expected_verdict": "Pass",
    },
    "weak_oop": {
        "topic": "Object-Oriented Programming",
        "question": "Explain dynamic method dispatch and how runtime polymorphism is achieved in Java.",
        "answer": "Polymorphism means many forms. You have method overloading which is compile time and overriding.",
        "expected_max_score": 6.0,
        "expected_verdict": "Weak",
    },
    "strong_concurrency": {
        "topic": "Concurrency & Multithreading",
        "question": "How does ConcurrentHashMap achieve thread safety without blocking all read operations?",
        "answer": (
            "In Java 8+, ConcurrentHashMap avoids global table locks by using bucket-level synchronization. "
            "Retrieval operations (get) generally do not block and proceed using volatile reads on Node's val and next pointers. "
            "Updates use Compare-And-Swap (CAS) for inserting into empty buckets and synchronize only on the head node of the specific "
            "bin when a collision occurs, maintaining high concurrency and O(1) lookup."
        ),
        "expected_min_score": 8.5,
        "expected_verdict": "Pass",
    },
    "weak_concurrency": {
        "topic": "Concurrency & Multithreading",
        "question": "How does ConcurrentHashMap achieve thread safety without blocking all read operations?",
        "answer": "It is faster than Hashtable because it uses threads and synchronized keyword.",
        "expected_max_score": 5.0,
        "expected_verdict": "Weak",
    },
    "empty_answer": {
        "topic": "Core Fundamentals",
        "question": "Explain JVM Heap versus Stack allocation.",
        "answer": "   ",
        "expected_max_score": 3.0,
        "expected_verdict": "Weak",
    },
    "code_snippet_answer": {
        "topic": "Algorithm Problem Solving",
        "question": "Write a method to detect a cycle in a singly linked list with O(1) extra space.",
        "answer": (
            "public boolean hasCycle(ListNode head) {\n"
            "    if (head == null || head.next == null) return false;\n"
            "    ListNode slow = head, fast = head;\n"
            "    while (fast != null && fast.next != null) {\n"
            "        slow = slow.next;\n"
            "        fast = fast.next.next;\n"
            "        if (slow == fast) return true;\n"
            "    }\n"
            "    return false;\n"
            "}"
        ),
        "expected_min_score": 8.5,
        "expected_verdict": "Pass",
    },
}

# ── Sample Placement Study Documents for Ingestion Testing ────────────────────

SAMPLE_DOCUMENTS: List[Dict[str, str]] = [
    {
        "filename": "sample_system_design_notes.txt",
        "content": (
            "=== SYSTEM DESIGN INTERVIEW GUIDE ===\n\n"
            "1. Horizontal vs Vertical Scaling: Horizontal scaling adds more machines into the pool of resources, "
            "while vertical scaling adds more power (CPU, RAM) to an existing machine.\n\n"
            "2. CAP Theorem: A distributed system can only provide two of Consistency, Availability, and Partition Tolerance.\n\n"
            "3. Caching Strategies: Cache-aside, Write-through, Write-behind, and Refresh-ahead. LRU eviction is commonly used."
        ),
    },
    {
        "filename": "sample_sql_optimization_guide.txt",
        "content": (
            "=== DATABASE & SQL PERFORMANCE TUNING ===\n\n"
            "1. Indexes: B-Tree indexes provide O(log N) lookup. Avoid indexing low-cardinality columns.\n\n"
            "2. EXPLAIN PLAN: Check for full table scans vs index range scans to optimize query latency.\n\n"
            "3. ACID Properties: Atomicity, Consistency, Isolation, and Durability ensure reliable transaction processing."
        ),
    },
]
