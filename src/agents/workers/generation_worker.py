"""Worker that generates final answer with citations (Task 10 helpers)."""

from __future__ import annotations

import os

from src.agents.messages import WorkerRequest, WorkerResponse
from src.task10_generation import (
    LLM_MODEL,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    format_context,
    reorder_for_llm,
)


def run_generation_worker(request: WorkerRequest) -> WorkerResponse:
    chunks = list(request.context.get("chunks", []))
    if not chunks:
        return WorkerResponse(
            worker="generation_worker",
            status="skipped",
            payload={
                "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
                "sources": [],
                "retrieval_source": "none",
            },
            reasoning="No chunks available, skip generation.",
        )

    retrieval_source = str(request.context.get("retrieval_source", "hybrid"))
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {request.query}"

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return WorkerResponse(
            worker="generation_worker",
            status="ok",
            payload={
                "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có (thiếu OPENAI_API_KEY).",
                "sources": chunks,
                "retrieval_source": retrieval_source,
            },
            reasoning="OPENAI_API_KEY missing; returned safe fallback answer.",
        )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content or ""
        status = "ok"
        reasoning = "Generated answer with citation constraints from context."
    except Exception as exc:
        answer = f"Tôi không thể xác minh thông tin này từ nguồn hiện có. (Lỗi LLM: {exc})"
        status = "error"
        reasoning = "LLM call failed; returned safe fallback answer."

    return WorkerResponse(
        worker="generation_worker",
        status=status,
        payload={
            "answer": answer,
            "sources": chunks,
            "retrieval_source": retrieval_source,
        },
        reasoning=reasoning,
    )
