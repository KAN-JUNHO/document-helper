from typing import Set
import time

import streamlit as st
from streamlit_chat import message
from backend.core import run_llm_OPENAI

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
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"
    return sources_string


st.header("LangChain🦜🔗 Helper Bot")

# Initialize session states if they don't exist
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "source_documents" not in st.session_state:
    st.session_state["source_documents"] = []
if "show_k" not in st.session_state:
    st.session_state["show_k"] = True
if "show_answer" not in st.session_state:
    st.session_state["show_answer"] = True

# 컬럼을 만들어 왼쪽에 설정 옵션들을 배치하고 오른쪽에 채팅 인터페이스를 배치
settings_column, chat_column = st.columns([1, 2])

with settings_column:
    st.subheader("Settings")
    search_type = st.selectbox("Select search type", ["mmr", "similarity","similarity_score_threshold","lambda_mult"])
    chunk_size = st.selectbox("Select chunk size", [100, 200, 500, 1000, 1800])
    chunk_overlap = st.selectbox("Select chunk overlap", [0, 10])

    k = st.text_input("k number here :", value=4)
    fetch_k = st.text_input("fetch_k number here:", value=20)
    similarity_score_threshold = st.text_input("similarity_score_threshold number here:", value=20)
    lambda_mult = st.text_input("lambda_mult float 0~1 here:", value=0.5)
    chain_type = st.selectbox("Select chainType", ["stuff", "map_reduce", "refine"])

    # 토글 생성
    st.session_state["show_k"] = st.checkbox("Show K value", value=st.session_state["show_k"])
    st.session_state["show_answer"] = st.checkbox("Show Answer", value=st.session_state["show_answer"])

with chat_column:
    # prompt 값을 세션 상태에서 가져오거나 기본값으로 초기화
    prompt = st.text_input("Prompt", value=st.session_state.get("prompt", ""), placeholder="Enter your message here...")
    st.session_state["prompt"] = prompt  # 사용자의 입력을 세션 상태에 저장

    search_kwargs = {
        "k": int(k),
        "fetch_k": int(fetch_k),
        "score_threshold": float(similarity_score_threshold),
        "lambda_mult": float(lambda_mult)
    }

    if st.button("Submit"):
        with st.spinner("Generating response..."):
            generated_response = run_llm_OPENAI(
                query=prompt,
                search_type=search_type,
                chat_history=st.session_state["chat_history"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                search_kwargs=search_kwargs,
                chain_type=chain_type,
            )
            formatted_response = f"{generated_response['answer']}"
            st.session_state.chat_history.append((prompt, generated_response["answer"]))
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)
            source_docs_list = generated_response["source_documents"][0 : int(k)]
            for doc in source_docs_list:
                st.session_state["source_documents"].append(doc.page_content)

        for idx, (generated_response, user_query) in enumerate(
                zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"])):
            unique_time = str(time.time()).replace('.', '')
            message(user_query, is_user=True, key=f"user_msg_{idx}_{unique_time}")

            if st.session_state["show_answer"]:
                message(generated_response, key=f"bot_msg_{idx}_{unique_time}")

            if st.session_state["show_k"]:
                for doc_idx, k in enumerate(st.session_state["source_documents"]):
                    light_red_background_message = f'''
                    <div style="background-color: rgba(255, 0, 0, 0.3); 
                                padding: 10px; 
                                border-radius: 5px; 
                                max-width: 80%; 
                                margin: 10px 0;">
                        반환된 문서 K : {k}
                    </div>
                    '''
                    st.markdown(light_red_background_message, unsafe_allow_html=True)
