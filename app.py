"""
RAG Chatbot — Pháp luật & Tin tức Ma tuý Việt Nam

Chạy:
    streamlit run app.py

Stack: Streamlit → Task 9 (Retrieval) → Task 10 (Generation) → Display
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot — Pháp luật Ma tuý",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.source-box {
    background:#f0f4ff; border-radius:8px; padding:12px 16px;
    margin:4px 0; border-left:3px solid #4f6ef7; font-size:0.85rem;
}
.source-label { font-weight:600; color:#4f6ef7; }
.badge-hybrid { background:#e8f4e8; color:#1a7a1a; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
.badge-pageindex { background:#fff3e0; color:#e65100; padding:2px 8px; border-radius:12px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─── Lazy-load pipeline (cached) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang khởi động RAG pipeline…")
def load_pipeline():
    from src.agents.run import run_pipeline_dict
    return run_pipeline_dict


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ RAG Chatbot")
    st.caption("Pháp luật & Tin tức Ma tuý Việt Nam")
    st.divider()

    st.subheader("⚙️ Cấu hình")
    top_k = st.slider("Số chunks retrieval (top_k)", 2, 10, 5)
    use_reranking = st.toggle("Bật Reranking (Jina)", value=True)
    show_sources = st.toggle("Hiển thị nguồn tài liệu", value=True)

    st.divider()
    st.subheader("🔍 Test nhanh")
    sample_queries = [
        "Hình phạt tội tàng trữ ma tuý theo pháp luật VN?",
        "Ca sĩ nào bị bắt vì liên quan ma tuý năm 2024?",
        "Quy trình cai nghiện bắt buộc theo luật 2025?",
        "Nghị định 28/2026 quy định gì về danh mục chất cấm?",
        "Hoài DJ bị xử như thế nào?",
    ]
    for q in sample_queries:
        if st.button(q, use_container_width=True, key=f"btn_{q[:20]}"):
            st.session_state.pending_query = q

    st.divider()
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.source_history = []
        st.session_state.trace_history = []
        st.rerun()

    st.caption("Nguồn dữ liệu: Luật PCMT 2025, Nghị định 163 & 28/2026, VnExpress, Ngôi sao")

# ─── Session state ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_history" not in st.session_state:
    st.session_state.source_history = []
if "trace_history" not in st.session_state:
    st.session_state.trace_history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ─── Main area ───────────────────────────────────────────────────────────────
st.title("⚖️ Chatbot Pháp luật & Tin tức Ma tuý")
st.caption(
    "Hỏi về quy định pháp luật hoặc tin tức nghệ sĩ liên quan tới ma tuý. "
    "Câu trả lời có **citation** từ nguồn tài liệu thực."
)

# Render conversation history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show sources for assistant messages
        if msg["role"] == "assistant" and show_sources and (i // 2) < len(st.session_state.source_history):
            sources = st.session_state.source_history[i // 2]
            if sources:
                with st.expander(f"📚 {len(sources)} nguồn tài liệu đã dùng", expanded=False):
                    for j, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "unknown").replace(".md","").replace(".docx","")
                        doc_type = meta.get("type", "")
                        score = src.get("score", 0)
                        retrieval = src.get("source", "hybrid")
                        badge = f'<span class="badge-hybrid">hybrid</span>' if retrieval == "hybrid" else f'<span class="badge-pageindex">pageindex</span>'
                        st.markdown(
                            f'<div class="source-box">'
                            f'<span class="source-label">[{j}] {source_name}</span> '
                            f'{badge} &nbsp; type: {doc_type} &nbsp; score: {score:.4f}<br/>'
                            f'<small>{src["content"][:200]}…</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
        if msg["role"] == "assistant" and (i // 2) < len(st.session_state.trace_history):
            trace_events = st.session_state.trace_history[i // 2]
            if trace_events:
                with st.expander("🧠 Reasoning flow (trace)", expanded=False):
                    for idx, event in enumerate(trace_events, 1):
                        metadata = event.get("metadata", {})
                        meta_text = (
                            f" | metadata: {metadata}" if metadata else ""
                        )
                        st.markdown(
                            f"{idx}. **{event.get('agent', 'unknown')}** → "
                            f"`{event.get('step', 'unknown')}` "
                            f"({event.get('duration_ms', 0)}ms)\n\n"
                            f"- Input: {event.get('input_summary', '')}\n"
                            f"- Output: {event.get('output_summary', '')}{meta_text}"
                        )

# Handle sample query buttons
if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    user_input = st.chat_input("Nhập câu hỏi về pháp luật ma tuý…")

if user_input:
    # Build conversation context for follow-up support
    history_context = ""
    if len(st.session_state.messages) >= 2:
        recent = st.session_state.messages[-4:]  # last 2 turns
        history_context = "\n".join(
            f"{'Người dùng' if m['role']=='user' else 'Trợ lý'}: {m['content']}"
            for m in recent
        )

    # Augment query with conversation context if follow-up
    augmented_query = user_input
    if history_context:
        augmented_query = f"[Lịch sử hội thoại:\n{history_context}\n]\nCâu hỏi mới: {user_input}"

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tổng hợp…"):
            try:
                run_pipeline = load_pipeline()
                result = run_pipeline(
                    augmented_query,
                    top_k=top_k,
                    use_reranking=use_reranking,
                )
                answer = result["answer"]
                sources = result["sources"]
                trace_events = result.get("trace", [])
            except Exception as e:
                answer = f"❌ Lỗi hệ thống: {e}"
                sources = []
                trace_events = []

        st.markdown(answer)

        # Show sources inline
        if show_sources and sources:
            with st.expander(f"📚 {len(sources)} nguồn tài liệu đã dùng", expanded=False):
                for j, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "unknown").replace(".md","").replace(".docx","")
                    doc_type = meta.get("type", "")
                    score = src.get("score", 0)
                    retrieval = src.get("source", "hybrid")
                    badge = f'<span class="badge-hybrid">hybrid</span>' if retrieval == "hybrid" else f'<span class="badge-pageindex">pageindex</span>'
                    st.markdown(
                        f'<div class="source-box">'
                        f'<span class="source-label">[{j}] {source_name}</span> '
                        f'{badge} &nbsp; type: {doc_type} &nbsp; score: {score:.4f}<br/>'
                        f'<small>{src["content"][:200]}…</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    # Store sources indexed by assistant turn index
    st.session_state.source_history.append(sources)
    st.session_state.trace_history.append(trace_events)
