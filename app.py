import streamlit as st

import api
import database


st.set_page_config(
    page_title="AI Model Finder",
    page_icon="🤖",
    layout="wide"
)

# 프로그램을 시작할 때 데이터베이스와 테이블을 생성합니다.
database.create_table()

TASK_OPTIONS = {
    "전체 작업": None,
    "문장 분류·감정 분석": "text-classification",
    "문장 요약": "summarization",
    "질문 답변": "question-answering",
    "이미지 분류": "image-classification",
    "이미지 생성": "text-to-image",
    "음성 인식": "automatic-speech-recognition",
    "글 생성": "text-generation"
}

PRIORITIES = ["높음", "중", "낮음"]


# -------------------- 공통 화면 --------------------
st.title("🤖 AI Model Finder")
st.caption("프로젝트 목적에 적합한 오픈소스 AI 모델을 탐색하고 관리하는 웹앱")

menu = st.sidebar.selectbox(
    "메뉴 선택",
    ["AI 모델 탐색", "내 모델 목록", "AI 모델 직접 등록"]
)


# -------------------- 1. 외부 AI 모델 검색 --------------------
if menu == "AI 모델 탐색":
    st.header("🔎 AI 모델 탐색")
    st.write("필요한 AI 작업을 선택하거나 모델명을 입력하여 Hugging Face 모델을 검색하세요.")

    with st.form("model_search_form"):
        purpose_name = st.selectbox("사용 목적", list(TASK_OPTIONS.keys()))
        keyword = st.text_input("모델 검색어", placeholder="예: bert, resnet, 텍스트분류, 이미지분류")
        st.caption("텍스트분류, 문장요약, 질문답변, 이미지분류, 이미지생성, 음성인식, 글생성과 같이 한국어 작업명을 입력해도 검색할 수 있습니다.")
        search_button = st.form_submit_button("모델 검색")

    if search_button:
        keyword = keyword.strip()
        task = TASK_OPTIONS[purpose_name]

        detected_task = api.detect_task_keyword(keyword)

        if detected_task:
            detected_task_name = api.get_task_display_name(detected_task)
            if task is None or task == detected_task:
                st.info(
                    "한국어 검색어 '" + keyword + "'을(를) '" + detected_task_name
                    + "' 작업 분류로 인식하여 검색합니다."
                )
            else:
                st.warning(
                    "입력한 검색어는 '" + detected_task_name
                    + "' 작업이지만, 사용 목적에서 선택한 작업 분류를 기준으로 검색합니다."
                )

        if keyword == "" and task is None:
            st.session_state["model_search_results"] = []
            st.warning("검색어 또는 작업 분류를 선택해주세요.")
        else:
            try:
                results = api.search_external_models(keyword, task, limit=10)
                st.session_state["model_search_results"] = results
                st.session_state["model_search_error"] = ""
            except RuntimeError as error:
                st.session_state["model_search_results"] = []
                st.session_state["model_search_error"] = str(error)

    error_message = st.session_state.get("model_search_error", "")
    if error_message:
        st.error(error_message)

    results = st.session_state.get("model_search_results", [])

    if results:
        st.subheader("검색 결과")
        st.caption("다운로드 수가 많은 순서로 최대 10개의 모델을 표시합니다.")

        for index, model in enumerate(results):
            with st.container(border=True):
                st.subheader(model["model_name"])
                col1, col2 = st.columns(2)
                col1.write("**작업 분류:** " + str(api.get_task_display_name(model["task"])))
                col2.write("**다운로드 수:** " + format(model["downloads"], ","))
                st.link_button("Hugging Face 모델 페이지 열기", model["model_url"])

                with st.form("save_model_form_" + str(index)):
                    idea = st.text_input(
                        "내 프로젝트 적용 아이디어",
                        placeholder="예: 사진 속 사물을 자동으로 분류하는 웹사이트"
                    )
                    priority = st.selectbox(
                        "우선순위",
                        PRIORITIES,
                        index=1,
                        key="search_priority_" + str(index)
                    )
                    save_button = st.form_submit_button("내 모델 목록에 저장")

                if save_button:
                    idea = idea.strip()
                    if idea == "":
                        st.warning("적용 아이디어를 입력해주세요.")
                    else:
                        changed = database.add_model(
                            model["model_id"],
                            model["model_name"],
                            model["task"],
                            model["downloads"],
                            idea,
                            priority,
                            model["model_url"],
                            "Hugging Face API"
                        )
                        if changed:
                            st.success("모델이 내 목록에 저장되었습니다.")
                        else:
                            st.warning("이미 저장된 모델입니다.")

    elif st.session_state.get("model_search_results") is not None and not error_message:
        if "model_search_results" in st.session_state:
            st.info("검색 결과가 없습니다.")


# -------------------- 2. 저장한 모델 조회 및 관리 --------------------
elif menu == "내 모델 목록":
    st.header("📚 내 AI 모델 목록")
    st.write("등록한 모델을 확인하고 모델명으로 검색하거나 우선순위를 변경할 수 있습니다.")

    keyword = st.text_input("저장한 모델 검색", placeholder="모델명을 입력하세요.")
    models = database.search_models(keyword)

    if len(models) == 0:
        if keyword.strip() == "":
            st.info("등록된 모델이 없습니다.")
        else:
            st.info("검색 결과가 없습니다.")
    else:
        st.caption("총 " + str(len(models)) + "개의 모델이 검색되었습니다.")

        for model in models:
            # 데이터 순서: id, model_id, model_name, task, downloads,
            # use_idea, priority, model_url, source
            with st.container(border=True):
                st.subheader(model[2])
                col1, col2, col3 = st.columns(3)
                col1.write("**작업 분류**\n\n" + str(api.get_task_display_name(model[3])))
                col2.write("**다운로드 수**\n\n" + format(model[4], ","))
                col3.write("**우선순위**\n\n" + str(model[6]))
                st.write("**적용 아이디어:** " + model[5])
                st.caption("등록 방식: " + str(model[8]))

                if model[7]:
                    st.link_button("모델 페이지 열기", model[7])

                col_update, col_delete = st.columns(2)
                with col_update:
                    with st.form("priority_form_" + str(model[0])):
                        new_priority = st.selectbox(
                            "우선순위 변경",
                            PRIORITIES,
                            index=PRIORITIES.index(model[6]) if model[6] in PRIORITIES else 1,
                            key="priority_" + str(model[0])
                        )
                        update_button = st.form_submit_button("우선순위 저장")

                    if update_button:
                        database.update_priority(model[0], new_priority)
                        st.success("우선순위가 변경되었습니다.")
                        st.rerun()

                with col_delete:
                    with st.form("delete_form_" + str(model[0])):
                        st.write("모델 삭제")
                        delete_button = st.form_submit_button("삭제")

                    if delete_button:
                        database.delete_model(model[0])
                        st.success("모델이 삭제되었습니다.")
                        st.rerun()


# -------------------- 3. 모델 직접 등록 --------------------
else:
    st.header("➕ AI 모델 직접 등록")
    st.write("Hugging Face 검색 결과가 아니더라도 원하는 AI 모델을 직접 등록할 수 있습니다.")

    task_choice = st.selectbox(
        "작업 분류",
        ["text-classification", "summarization", "question-answering",
         "image-classification", "text-to-image",
         "automatic-speech-recognition", "text-generation", "직접 입력"]
    )

    if task_choice == "직접 입력":
        task_name = st.text_input("작업 분류 직접 입력", placeholder="작업 분류를 입력하세요.")
    else:
        task_name = task_choice

    with st.form("add_model_form"):
        model_name = st.text_input("모델명", placeholder="예: my-custom-model")
        use_idea = st.text_area(
            "적용 아이디어",
            placeholder="이 모델을 어떤 프로젝트에 활용할 것인지 작성하세요."
        )
        register_button = st.form_submit_button("모델 등록")

    if register_button:
        model_name = model_name.strip()
        use_idea = use_idea.strip()
        task_name = task_name.strip()

        if model_name == "" or use_idea == "":
            st.error("모델명과 적용 아이디어를 모두 입력해주세요.")
        elif task_name == "":
            st.error("작업 분류를 입력해주세요.")
        else:
            # 직접 등록한 모델은 외부 모델 ID 대신 manual:로 시작하는 ID를 사용합니다.
            manual_id = "manual:" + model_name.lower()
            changed = database.add_model(
                manual_id,
                model_name,
                task_name,
                0,
                use_idea,
                "중",
                "",
                "직접 등록"
            )

            if changed:
                st.success("AI 모델이 등록되었습니다.")
            else:
                st.warning("같은 이름의 모델이 이미 등록되어 있습니다.")


st.sidebar.divider()
st.sidebar.caption("AI Model Finder | Python · Streamlit · SQLite")
