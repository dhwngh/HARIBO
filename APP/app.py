"""사용자 목적 기반 AI 모델 탐색·관리 Streamlit 앱."""

import sqlite3

import streamlit as st

import api
import database


st.set_page_config(
    page_title="AI Model Finder",
    page_icon="🤖",
    layout="wide",
)

try:
    database.create_table()
except sqlite3.Error as error:
    st.error("데이터베이스를 준비하지 못했습니다. 관리자에게 오류 내용을 확인해주세요.")
    st.exception(error)
    st.stop()

TASK_OPTIONS = {
    "전체 작업": None,
    "문장 분류·감정 분석": "text-classification",
    "문장 요약": "summarization",
    "질문 답변": "question-answering",
    "이미지 분류": "image-classification",
    "이미지 생성": "text-to-image",
    "음성 인식": "automatic-speech-recognition",
    "글 생성": "text-generation",
}
TASK_LABEL_TO_TAG = {label: tag for label, tag in TASK_OPTIONS.items() if tag}
PRIORITIES = ["높음", "중", "낮음"]


# -------------------- 공통 화면 --------------------
st.title("🤖 AI Model Finder")
st.caption("프로젝트 목적에 맞는 오픈소스 AI 모델을 탐색하고 개인 목록으로 관리하는 웹앱")

menu = st.sidebar.selectbox(
    "메뉴 선택",
    ["AI 모델 탐색", "내 모델 목록", "AI 모델 직접 등록"],
)
st.sidebar.divider()
st.sidebar.caption("검색 결과는 Hugging Face 공개 모델 정보를 사용합니다.")
st.sidebar.caption("다운로드 수는 인기도를 참고하는 지표이며 모델 성능을 보장하지 않습니다.")


# -------------------- 1. 외부 AI 모델 검색 --------------------
if menu == "AI 모델 탐색":
    st.header("🔎 AI 모델 탐색")
    st.write("필요한 작업 분류를 선택하거나 모델명을 입력하여 Hugging Face 모델을 검색하세요.")

    with st.form("model_search_form"):
        purpose_name = st.selectbox("사용 목적", list(TASK_OPTIONS.keys()))
        keyword = st.text_input(
            "모델 검색어 또는 한국어 작업명",
            placeholder="예: bert, resnet, 텍스트분류, 이미지분류",
        )
        st.caption(
            "한국어 작업명 예시: 텍스트 분류, 감정 분석, 문장 요약, 질문 답변, "
            "이미지 분류, 이미지 생성, 음성 인식, 글 생성"
        )
        search_button = st.form_submit_button("모델 검색", type="primary")

    if "model_search_status" not in st.session_state:
        st.session_state["model_search_status"] = "not_started"
        st.session_state["model_search_results"] = []
        st.session_state["model_search_error"] = ""
        st.session_state["model_search_info"] = ""

    if search_button:
        # 새 검색을 시작할 때 이전 결과와 오류를 먼저 지워 이전 결과가 남지 않게 합니다.
        st.session_state["model_search_results"] = []
        st.session_state["model_search_error"] = ""
        st.session_state["model_search_info"] = ""
        st.session_state["model_search_status"] = "not_started"

        keyword = keyword.strip()
        selected_task = TASK_OPTIONS[purpose_name]
        detected_task = api.detect_task_keyword(keyword)

        if detected_task and selected_task is None:
            selected_task = detected_task
            st.session_state["model_search_info"] = (
                "한국어 작업명 '" + keyword + "'을(를) '"
                + api.get_task_display_name(detected_task)
                + "' 작업 분류로 변환하여 검색합니다."
            )
            keyword = ""
        elif detected_task and selected_task and detected_task != selected_task:
            st.session_state["model_search_info"] = (
                "입력한 한국어 작업명과 선택한 사용 목적이 다릅니다. "
                "선택한 사용 목적을 우선하여 검색합니다."
            )
            keyword = ""

        if not keyword and selected_task is None:
            st.session_state["model_search_status"] = "empty_input"
        else:
            try:
                results = api.search_external_models(keyword, selected_task, limit=10)
                st.session_state["model_search_results"] = results
                st.session_state["model_search_status"] = "results" if results else "no_results"
            except RuntimeError as error:
                st.session_state["model_search_error"] = str(error)
                st.session_state["model_search_status"] = "error"

    if st.session_state.get("model_search_info"):
        st.info(st.session_state["model_search_info"])

    search_status = st.session_state.get("model_search_status", "not_started")
    if search_status == "empty_input":
        st.warning("검색어를 입력해주세요.")
    elif search_status == "error":
        st.error(st.session_state.get("model_search_error", "검색 중 오류가 발생했습니다."))
    elif search_status == "no_results":
        st.info("검색 결과가 없습니다.")

    results = st.session_state.get("model_search_results", [])
    if search_status == "results" and results:
        st.subheader("검색 결과")
        st.caption("다운로드 수가 많은 순서로 최대 10개의 모델을 표시합니다.")
        st.caption("다운로드 수는 인기도 참고 자료입니다. 모델의 정확도나 성능 순위와 동일하지 않습니다.")

        for index, model in enumerate(results):
            with st.container(border=True):
                st.subheader(model["model_name"])
                col_task, col_downloads = st.columns(2)
                col_task.write("**작업 분류:** " + api.get_task_display_name(model["task"]))
                col_downloads.write("**다운로드 수:** " + format(model["downloads"], ","))
                st.link_button("Hugging Face 모델 페이지 열기", model["model_url"])

                with st.form("save_model_form_" + str(index)):
                    idea = st.text_input(
                        "내 프로젝트 적용 아이디어",
                        placeholder="예: 사진 속 사물을 자동으로 분류하는 웹사이트",
                    )
                    priority = st.selectbox(
                        "우선순위",
                        PRIORITIES,
                        index=1,
                        key="search_priority_" + str(index),
                    )
                    save_button = st.form_submit_button("내 모델 목록에 저장")

                if save_button:
                    idea = idea.strip()
                    if not idea:
                        st.warning("적용 아이디어를 입력해주세요.")
                    else:
                        try:
                            changed = database.add_model(
                                model["model_id"],
                                model["model_name"],
                                model["task"],
                                model["downloads"],
                                idea,
                                priority,
                                model["model_url"],
                                "Hugging Face API",
                            )
                            if changed:
                                st.success("모델이 내 목록에 저장되었습니다.")
                            else:
                                st.warning("이미 내 목록에 저장된 모델입니다.")
                        except (ValueError, sqlite3.Error) as error:
                            st.error("모델을 저장하지 못했습니다: " + str(error))


# -------------------- 2. 저장한 모델 조회 및 관리 --------------------
elif menu == "내 모델 목록":
    st.header("📚 내 AI 모델 목록")
    st.write("저장한 모델을 확인하고 모델명으로 검색하거나 우선순위를 변경·삭제할 수 있습니다.")

    keyword = st.text_input("저장한 모델 검색", placeholder="모델명을 입력하세요.")
    priority_filter = st.selectbox("우선순위 필터", ["전체", "높음", "중", "낮음"])

    try:
        models = database.search_models(keyword)
    except sqlite3.Error as error:
        st.error("저장된 모델을 불러오지 못했습니다: " + str(error))
        models = []

    if priority_filter != "전체":
        models = [model for model in models if model[6] == priority_filter]

    all_models = database.get_all_models()
    high_priority_count = sum(1 for model in all_models if model[6] == "높음")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("전체 저장 모델", len(all_models))
    metric2.metric("높은 우선순위", high_priority_count)
    metric3.metric("현재 검색 결과", len(models))

    if not models:
        if not all_models:
            st.info("등록된 모델이 없습니다.")
        elif keyword.strip() or priority_filter != "전체":
            st.info("검색 조건에 맞는 모델이 없습니다.")
        else:
            st.info("등록된 모델이 없습니다.")
    else:
        for model in models:
            # 데이터 순서: id, model_id, model_name, task, downloads,
            # use_idea, priority, model_url, source
            with st.container(border=True):
                st.subheader(model[2])
                col_task, col_downloads, col_priority = st.columns(3)
                col_task.write("**작업 분류**\n\n" + api.get_task_display_name(model[3]))
                col_downloads.write("**다운로드 수**\n\n" + format(model[4], ","))
                col_priority.write("**우선순위**\n\n" + model[6])
                st.write("**적용 아이디어:** " + model[5])
                st.caption("등록 방식: " + str(model[8]))

                if model[7]:
                    st.link_button("모델 페이지 열기", model[7])

                col_update, col_delete = st.columns(2)
                with col_update:
                    with st.form("priority_form_" + str(model[0])):
                        current_priority_index = PRIORITIES.index(model[6]) if model[6] in PRIORITIES else 1
                        new_priority = st.selectbox(
                            "우선순위 변경",
                            PRIORITIES,
                            index=current_priority_index,
                            key="priority_" + str(model[0]),
                        )
                        update_button = st.form_submit_button("우선순위 저장")
                    if update_button:
                        try:
                            database.update_priority(model[0], new_priority)
                            st.success("우선순위가 변경되었습니다.")
                            st.rerun()
                        except (ValueError, sqlite3.Error) as error:
                            st.error("우선순위를 변경하지 못했습니다: " + str(error))

                with col_delete:
                    with st.form("delete_form_" + str(model[0])):
                        st.write("모델 삭제")
                        delete_button = st.form_submit_button("삭제")
                    if delete_button:
                        try:
                            database.delete_model(model[0])
                            st.success("모델이 삭제되었습니다.")
                            st.rerun()
                        except sqlite3.Error as error:
                            st.error("모델을 삭제하지 못했습니다: " + str(error))


# -------------------- 3. 모델 직접 등록 --------------------
else:
    st.header("➕ AI 모델 직접 등록")
    st.write("외부 검색 결과가 아니더라도 원하는 AI 모델을 직접 등록할 수 있습니다.")

    task_labels = list(TASK_LABEL_TO_TAG.keys()) + ["기타 (직접 입력)"]
    task_choice = st.selectbox("작업 분류", task_labels)

    if task_choice == "기타 (직접 입력)":
        task_name = st.text_input("작업 분류 직접 입력", placeholder="예: 객체 탐지")
    else:
        task_name = TASK_LABEL_TO_TAG[task_choice]

    with st.form("add_model_form"):
        model_name = st.text_input("모델명", placeholder="예: my-custom-model")
        use_idea = st.text_area(
            "적용 아이디어",
            placeholder="이 모델을 어떤 프로젝트에 활용할 것인지 작성하세요.",
        )
        register_button = st.form_submit_button("모델 등록", type="primary")

    if register_button:
        model_name = model_name.strip()
        use_idea = use_idea.strip()
        task_name = task_name.strip()

        if not model_name or not use_idea:
            st.error("모델명과 적용 아이디어를 모두 입력해주세요.")
        elif not task_name:
            st.error("작업 분류를 입력해주세요.")
        else:
            manual_id = "manual:" + model_name.casefold()
            try:
                changed = database.add_model(
                    manual_id,
                    model_name,
                    task_name,
                    0,
                    use_idea,
                    "중",
                    "",
                    "직접 등록",
                )
                if changed:
                    st.success("AI 모델이 등록되었습니다. 다운로드 수는 0, 우선순위는 '중'으로 저장했습니다.")
                else:
                    st.warning("같은 이름의 모델이 이미 등록되어 있습니다.")
            except (ValueError, sqlite3.Error) as error:
                st.error("모델을 등록하지 못했습니다: " + str(error))


st.sidebar.divider()
st.sidebar.caption("AI Model Finder | Python · Streamlit · SQLite · Hugging Face API")
