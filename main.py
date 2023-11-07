import json
import os
import shutil
import time
from typing import Set
from uuid import uuid4

import pandas as pd
import streamlit as st
from streamlit_chat import message

from backend.core import run_llm_OPENAI
from db.DatabaseManager import DatabaseManager
from ingestion import chunk_make

# CSS for styling
st.markdown(
    """
    <style>
        .block-container {
            max-width: 80%;
        }
    </style>
""",
    unsafe_allow_html=True,
)


def create_sources_string(source_urls: Set[str]) -> str:
    """Generate a formatted string of sources."""
    sources_list = sorted(list(source_urls))
    return "\n".join([f"{i + 1}. {source}" for i, source in enumerate(sources_list)])


# Header
st.header("LangChain🦜🔗 Helper Bot")

# Session states initialization
for state, default_value in [
    ("user_prompt_history", []),
    ("chat_answers_history", []),
    ("chat_history", []),
    ("source_documents", []),
    ("show_k", True),
    ("show_answer", True),
]:
    st.session_state.setdefault(state, default_value)

if "db_manager" not in st.session_state:
    st.session_state["db_manager"] = DatabaseManager("chat_messages.db")
if "show_chat_history" not in st.session_state:
    st.session_state.show_chat_history = False
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid4())

# Layout: Settings and Chat columns
conversation_column, settings_column, chat_column = st.columns([1, 2, 3])


def save_chat_history_to_db():
    session_id = st.session_state["session_id"]
    db_manager = st.session_state["db_manager"]

    if len(st.session_state["user_prompt_history"]) > 1:
        user_query = st.session_state["user_prompt_history"][-1]
        resp = st.session_state["chat_answers_history"][-1]
        source_docs_json = st.session_state["source_documents"][-1]
        # source_docs 리스트를 JSON 문자열로 직렬화합니다.
        source_docs = json.dumps(
            [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in source_docs_json
            ],
            ensure_ascii=False,
        )
        fetch_k = search_kwargs.get("fetch_k", None)
        lambda_mult = search_kwargs.get("lambda_mult", None)
        k = search_kwargs.get("k", None)
        score_threshold = search_kwargs.get("score_threshold", None)

        db_manager.insert_message(
            user_query,
            k,
            fetch_k,
            lambda_mult,
            score_threshold,
            resp,
            source_docs,
            session_id,
        )
    else:
        for user_query, resp, source_docs_json in zip(
            st.session_state["user_prompt_history"],
            st.session_state["chat_answers_history"],
            st.session_state["source_documents"],
        ):
            # source_docs 리스트를 JSON 문자열로 직렬화합니다.
            source_docs = json.dumps(
                [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in source_docs_json
                ],
                ensure_ascii=False,
            )

            fetch_k = search_kwargs.get("fetch_k", None)
            lambda_mult = search_kwargs.get("lambda_mult", None)
            k = search_kwargs.get("k", None)
            score_threshold = search_kwargs.get("score_threshold", None)

            db_manager.insert_message(
                user_query,
                k,
                fetch_k,
                lambda_mult,
                score_threshold,
                resp,
                source_docs,
                session_id,
            )


def get_messages_by_session():
    db_manager = st.session_state["db_manager"]
    sessions = db_manager.fetch_sessions()

    # 세션 데이터를 보여주는 컨테이너를 만듭니다.
    with st.container():
        # 세션 목록에서 각 세션에 대해 버튼을 만듭니다.
        for user_query, session_id in sessions:
            # 세션 상태 관리를 위한 키를 생성합니다.
            session_state_key = f"clicked_{session_id}"

            # 세션 상태가 저장되어 있지 않으면 기본값으로 False를 설정합니다.
            if session_state_key not in st.session_state:
                st.session_state[session_state_key] = False

            # 버튼을 클릭했을 때의 동작을 정의합니다.
            if st.button(
                f"대화 질문: {user_query}\n\nsession: {session_id}", key=session_id
            ):
                # 버튼 상태를 토글합니다.
                st.session_state[session_state_key] = not st.session_state[
                    session_state_key
                ]

            # 세션 상태에 따라 메시지를 표시하거나 숨깁니다.
            if st.session_state[session_state_key]:
                # 해당 세션 ID의 메시지를 데이터베이스에서 가져옵니다.
                messages = db_manager.fetch_messages_by_session(session_id)
                # 메시지를 보여주는 또 다른 컨테이너를 만듭니다.
                with st.container():
                    # 가져온 메시지를 표시합니다.
                    for message in messages:
                        st.text_area(
                            f"Message ID: {message[0]}",
                            f"User: {message[1]}\nBot: {message[2]}",
                            height=100,
                        )
                        st.markdown("---")


# 채팅 기록을 위한 자리 표시자
chat_history_placeholder = st.empty()


def load_chat_history():
    db_manager = st.session_state["db_manager"]
    # DB 매니저에서 모든 메시지를 가져옵니다.
    all_messages = db_manager.fetch_all_messages()

    # 데이터를 판다스 데이터프레임으로 변환합니다.
    df_messages = pd.DataFrame(
        all_messages,
        columns=[
            "ID",
            "Session ID",
            "User Query",
            "Response",
            "Source Docs",
            "Timestamp",
        ],
    )

    # Streamlit에서 데이터프레임을 테이블로 보여줍니다.
    st.dataframe(df_messages)


def clear_chat_history():
    chat_history_placeholder.empty()


# Settings Column
with settings_column:
    st.subheader("Settings")

    uploaded_file = st.file_uploader("Upload a DOCX file", type="docx")

    path = "langchain-docs/langchain.readthedocs.io/en/latest/fairy_tails/"

    if uploaded_file:
        st.write("Uploaded file:", uploaded_file.name)
        # 파일 이름에서 확장자를 제거합니다.
        file_name_without_extension, _ = os.path.splitext(uploaded_file.name)
        # 저장할 경로를 지정합니다.
        target_path = os.path.join(path, file_name_without_extension)

        # 업로드한 파일을 해당 경로에 저장합니다.
        with open(target_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
            chunk_make(file_name_without_extension)

    # selected_files를 초기화합니다. 만약 session_state에 이미 존재하면 기존 값을 사용합니다.
    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []

    if path:
        try:
            files = os.listdir(path)
            # 파일 리스트를 순회하면서 각 파일에 대한 체크박스를 생성합니다.
            for file in files:
                file_name_without_extension, _ = os.path.splitext(file)
                # 체크박스 위젯을 생성하고, 현재 상태(is_checked)를 얻습니다.
                is_checked = st.checkbox(
                    f"{file_name_without_extension}", key=file_name_without_extension
                )

                # 체크박스가 선택되었는지 여부에 따라 selected_files 리스트를 업데이트합니다.
                if is_checked:
                    if (
                        file_name_without_extension
                        not in st.session_state["selected_files"]
                    ):
                        st.session_state["selected_files"].append(
                            file_name_without_extension
                        )
                else:
                    if (
                        file_name_without_extension
                        in st.session_state["selected_files"]
                    ):
                        st.session_state["selected_files"].remove(
                            file_name_without_extension
                        )
        except Exception as e:
            st.write(f"An error occurred: {e}")

    search_type = st.selectbox(
        "Select search type", ["mmr", "similarity", "similarity_score_threshold"]
    )
    chunk_size = st.selectbox("Select chunk size", [100, 200, 500, 1000, 1800])
    chunk_overlap = st.selectbox("Select chunk overlap", [0, 10])

    search_kwargs = {}
    k = st.text_input("k 변수 입력:", value="4")
    search_kwargs["k"] = int(k)

    if search_type == "similarity_score_threshold":
        score_threshold = st.text_input(
            "similarity_score_threshold 변수 입력:", value="0.8"
        )
        search_kwargs["score_threshold"] = float(score_threshold)

    elif search_type == "mmr":
        fetch_k = st.text_input("fetch_k 변수 입력:", value="20")
        lambda_mult = st.text_input("lambda_mult float 0~1 변수입력:", value="0.5")
        search_kwargs.update(
            {"fetch_k": int(fetch_k), "lambda_mult": float(lambda_mult)}
        )
    elif search_type == "similarity":
        fetch_k = st.text_input("fetch_k 변수 입력:", value="20")
        search_kwargs.update({"fetch_k": int(fetch_k)})
    chain_type = st.selectbox("Select chainType", ["stuff", "map_reduce", "refine"])

    st.session_state["show_k"] = st.checkbox("K 값", value=st.session_state["show_k"])
    st.session_state["show_answer"] = st.checkbox(
        "응답", value=st.session_state["show_answer"]
    )

# Chat Column

# 애플리케이션 시작 시 데이터베이스 매니저 인스턴스 생성
db_manager = DatabaseManager("chat_messages.db")


with conversation_column:
    st.subheader("모든 세션 대화 목록")

    get_messages_by_session()

with chat_column:
    if st.button("대화 목록 불러오기/숨기기"):
        st.session_state.show_chat_history = not st.session_state.show_chat_history
        if st.session_state.show_chat_history:
            load_chat_history()
        else:
            clear_chat_history()

    prompt = st.text_input("Prompt", value="", placeholder="Enter your message here...")
    answer = []
    if st.button("Submit"):
        if prompt:  # prompt가 비어있지 않은 경우에만 처리합니다.
            with st.spinner("Generating response..."):
                responses = run_llm_OPENAI(
                    query=prompt,
                    search_type=search_type,
                    chat_history=st.session_state["chat_history"],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    search_kwargs=search_kwargs,
                    chain_type=chain_type,
                    selected_files=st.session_state["selected_files"],
                )

                # 응답을 반복하며 각각 처리합니다.
                for response in responses:
                    if "answer" in response:
                        # 중복된 prompt를 추가하지 않습니다.
                        # if prompt not in st.session_state["user_prompt_history"]:
                        st.session_state["user_prompt_history"].append(prompt)
                        st.session_state["chat_answers_history"].append(
                            response["answer"]
                        )

                        # 응답에 소스 문서가 포함되어 있다면 처리합니다.
                        if "source_documents" in response:
                            st.session_state["source_documents"].append(
                                response["source_documents"]
                            )
                        st.session_state["last_response"] = response["answer"]
                    save_chat_history_to_db()

    # Display messages
    if st.session_state.get("last_response"):
        unique_time = str(time.time()).replace(".", "")

        for idx, (user_query, resp, source_docs) in enumerate(
            zip(
                st.session_state["user_prompt_history"],
                st.session_state["chat_answers_history"],
                st.session_state["source_documents"],
            )
        ):
            message(user_query, is_user=True, key=f"user_msg_{idx}_{unique_time}")

            response_and_docs = resp  # 봇의 응답을 시작으로 합니다.

            # 문서 K 값이 표시되어야 하는 경우 응답에 추가합니다.
            if st.session_state["show_k"]:
                for k in st.session_state["source_documents"][idx]:
                    page_content = k.page_content
                    source = k.metadata["source"]
                    file_name = source.split("/")[-1]
                    st.markdown(
                        f"""
                          <div style="background-color: rgba(255, 0, 0, 0.3);
                                      padding: 10px;
                                      border-radius: 5px;
                                      max-width: 80%;
                                      margin: 10px 0;">
                              반환된 문서 K : {page_content}\n{file_name}
                          </div>
                          """,
                        unsafe_allow_html=True,
                    )

            # 응답이 표시되어야 하는 경우 메시지를 표시합니다.
            if st.session_state["show_answer"]:
                message(
                    response_and_docs, is_user=False, key=f"bot_msg_{idx}_{unique_time}"
                )

# 애플리케이션이 종료될 때 데이터베이스 연결을 닫습니다.
db_manager.close()
