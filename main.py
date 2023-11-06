from typing import Set
import time
import shutil

import os
import streamlit as st
from streamlit_chat import message
from backend.core import run_llm_OPENAI
from docx import Document

from ingestion import chunk_make

# CSS for styling
st.markdown(
    """
    <style>
        .block-container {
            max-width: 80%;
        }
    </style>
""", unsafe_allow_html=True
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
    ("show_answer", True)
]:
    st.session_state.setdefault(state, default_value)

# Layout: Settings and Chat columns
settings_column, chat_column = st.columns([1, 2])


def chat_up():
    # Display messages
    unique_time = str(time.time()).replace('.', '')
    # 사용자의 질문과 봇의 응답, 그리고 문서 K 값을 순서대로 표시합니다.
    for idx, (user_query, resp) in enumerate(
            zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"])):
        # 사용자의 질문을 표시합니다.
        message(user_query, is_user=True, key=f"user_msg_{idx}_{unique_time}")

        # 해당 질문에 대한 문서 K 값을 표시합니다.
        if st.session_state["show_k"]:

            # 문서 K 값이 리스트의 리스트라면 아래와 같이 접근합니다.
            # 여기서는 각 질문에 대한 문서 K 리스트를 가정합니다.
            for doc_k in st.session_state["source_documents"]:
                st.markdown(
                    f'''
                    <div style="background-color: rgba(255, 0, 0, 0.3);
                                padding: 10px;
                                border-radius: 5px;
                                max-width: 80%;
                                margin: 10px 0;">
                        반환된 문서 K : {doc_k}
                    </div>
                    ''', unsafe_allow_html=True
                )

        # 봇의 응답을 표시합니다.
        if st.session_state["show_answer"]:
            message(resp, is_user=False, key=f"bot_msg_{idx}_{unique_time}")


# Settings Column
with settings_column:
    st.subheader("Settings")

    uploaded_file = st.file_uploader("Upload a DOCX file", type="docx")

    path = "langchain-docs/langchain.readthedocs.io/en/latest/fairy_tails/"

    if uploaded_file:
        st.write("Uploaded file:", uploaded_file.name)

        # 저장할 경로를 지정합니다.
        target_path = os.path.join(path, uploaded_file.name)

        # 업로드한 파일을 해당 경로에 저장합니다.
        with open(target_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
            chunk_make(uploaded_file.name)

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
                is_checked = st.checkbox(f"{file_name_without_extension}", key=file_name_without_extension)

                # 체크박스가 선택되었는지 여부에 따라 selected_files 리스트를 업데이트합니다.
                if is_checked:
                    if file_name_without_extension not in st.session_state["selected_files"]:
                        st.session_state["selected_files"].append(file_name_without_extension)
                else:
                    if file_name_without_extension in st.session_state["selected_files"]:
                        st.session_state["selected_files"].remove(file_name_without_extension)
        except Exception as e:
            st.write(f"An error occurred: {e}")

    search_type = st.selectbox("Select search type", ["mmr", "similarity", "similarity_score_threshold"])
    chunk_size = st.selectbox("Select chunk size", [100, 200, 500, 1000, 1800])
    chunk_overlap = st.selectbox("Select chunk overlap", [0, 10])

    search_kwargs = {}
    k = st.text_input("k 변수 입력:", value="4")
    search_kwargs["k"] = int(k)

    if search_type == "similarity_score_threshold":
        score_threshold = st.text_input("similarity_score_threshold 변수 입력:", value="0.8")
        search_kwargs["score_threshold"] = float(score_threshold)

    elif search_type == "mmr":
        fetch_k = st.text_input("fetch_k 변수 입력:", value="20")
        lambda_mult = st.text_input("lambda_mult float 0~1 변수입력:", value="0.5")
        search_kwargs.update({
            "fetch_k": int(fetch_k),
            "lambda_mult": float(lambda_mult)
        })

    chain_type = st.selectbox("Select chainType", ["stuff", "map_reduce", "refine"])

    st.session_state["show_k"] = st.checkbox("K 값", value=st.session_state["show_k"])
    st.session_state["show_answer"] = st.checkbox("응답", value=st.session_state["show_answer"])

# Chat Column


with chat_column:
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
                    selected_files=st.session_state["selected_files"]
                )

                index = 0
                # 응답을 반복하며 각각 처리합니다.
                for response in responses:
                    if 'answer' in response:

                        # 중복된 prompt를 추가하지 않습니다.
                        # if prompt not in st.session_state["user_prompt_history"]:
                        st.session_state["user_prompt_history"].append(prompt)
                        st.session_state["chat_answers_history"].append(response["answer"])
                        # 응답에 소스 문서가 포함되어 있다면 처리합니다.
                        if 'source_documents' in response:
                            st.session_state["source_documents"].append(response["source_documents"])
                        st.session_state["last_response"] = response["answer"]
                    # chat_up()  # 모든 응답을 처리한 후에 한 번만 chat_up을 호출합니다.
                    # index += 1
    #
    # if st.session_state["show_k"] or st.session_state["show_answer"]:
    #     chat_up(len(responses)-1)

    # Display messages
    if st.session_state.get("last_response"):
        unique_time = str(time.time()).replace('.', '')

        for idx, (user_query, resp, source_docs) in enumerate(
                zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"],
                    st.session_state["source_documents"])):
            message(user_query, is_user=True, key=f"user_msg_{idx}_{unique_time}")

            response_and_docs = resp  # 봇의 응답을 시작으로 합니다.

            # 문서 K 값이 표시되어야 하는 경우 응답에 추가합니다.
            if st.session_state["show_k"]:
                for k in st.session_state["source_documents"][idx]:
                    st.markdown(
                        f'''
                          <div style="background-color: rgba(255, 0, 0, 0.3);
                                      padding: 10px;
                                      border-radius: 5px;
                                      max-width: 80%;
                                      margin: 10px 0;">
                              반환된 문서 K : {k}
                          </div>
                          ''', unsafe_allow_html=True
                    )


            # 응답이 표시되어야 하는 경우 메시지를 표시합니다.
            if st.session_state["show_answer"]:
                message(response_and_docs, is_user=False, key=f"bot_msg_{idx}_{unique_time}")

