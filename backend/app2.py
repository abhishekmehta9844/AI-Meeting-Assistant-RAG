import streamlit as st

from app.services.meeting_library import list_meetings
from app.services.rag_pipeline import build_rag_v2


st.set_page_config(
    page_title="Meeting Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_rag_for_meeting(persist_dir: str, collection_name: str):
    return build_rag_v2(persist_dir=persist_dir, collection_name=collection_name)


def build_conversational_query(messages, new_question, max_turns):
    history = []
    turns = 0

    for msg in reversed(messages):
        if msg["role"] == "user":
            history.append(f"User: {msg['content']}")
            turns += 1
        elif msg["role"] == "assistant":
            history.append(f"Assistant: {msg['content']}")

        if turns >= max_turns:
            break

    history.reverse()
    if not history:
        return new_question

    return "Conversation so far:\n" + "\n".join(history) + "\n\nCurrent question:\n" + new_question


def format_meeting_date(raw_value: str) -> str:
    if not raw_value:
        return "Unknown date"
    return raw_value.replace("T", " ")[:19]


meetings = list_meetings()
st.title("Meeting Chat")
st.caption("Choose a meeting and chat only with that meeting's transcript.")

st.sidebar.header("Meetings")
st.sidebar.caption("Pick a meeting to open its private chat workspace.")

if not meetings:
    st.info("No meetings found yet. Run the meeting bot first to create one.")
    st.stop()

meeting_options = {
    f"{meeting.get('title', meeting['meeting_id'])} | {format_meeting_date(meeting.get('started_at', ''))[:16]}": meeting
    for meeting in meetings
}
selected_label = st.sidebar.selectbox("Meeting", list(meeting_options.keys()), label_visibility="collapsed")
selected_meeting = meeting_options[selected_label]

selected_meeting_id = selected_meeting["meeting_id"]
if st.session_state.get("active_meeting_id") != selected_meeting_id:
    st.session_state.active_meeting_id = selected_meeting_id
    st.session_state.messages = []

with st.sidebar.expander("Settings", expanded=False):
    MAX_HISTORY_TURNS = st.slider("Conversation memory", 1, 10, 4, help="How much recent chat context the assistant remembers.")
    RETRIEVE_K = st.slider("Search breadth", 2, 30, 8, help="How many transcript sections are considered before the answer is narrowed down.")
    FINAL_K = st.slider("Answer focus", 1, 8, 4, help="How many transcript sections are finally used to answer.")
    MIN_SCORE = st.slider("Strictness", 0.0, 1.0, 0.2, help="Higher values make answers more selective and conservative.")
    SHOW_CONTEXT = st.checkbox("Show retrieved transcript excerpts", False)

if st.sidebar.button("Clear current chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

rag = load_rag_for_meeting(
    selected_meeting["vector_store_dir"],
    selected_meeting["collection_name"],
)

st.subheader(selected_meeting.get("title", selected_meeting_id))
st.caption(f"Date: {format_meeting_date(selected_meeting.get('started_at', ''))}")
st.caption(f"Transcript: {selected_meeting.get('transcript_file', 'Unknown')}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about this meeting...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    conversational_query = build_conversational_query(
        st.session_state.messages[:-1],
        user_input,
        MAX_HISTORY_TURNS,
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag(
                conversational_query,
                retrieve_k=RETRIEVE_K,
                final_k=FINAL_K,
                min_score=MIN_SCORE,
                return_context=SHOW_CONTEXT,
            )
            answer = result.get("answer", "I don't know.")
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    if result.get("sources"):
        with st.expander("Sources"):
            for source in result["sources"]:
                st.markdown(
                    f"- **{source.get('source', 'unknown')}** "
                    f"(score: `{source.get('score', 0):.2f}`)"
                )

    if SHOW_CONTEXT and result.get("context"):
        with st.expander("Retrieved Context"):
            st.text(result["context"])
