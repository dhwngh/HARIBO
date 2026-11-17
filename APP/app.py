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


# -------------------- 화면 디자인 --------------------
st.markdown(
    """
    <style>
    /* 전체 화면 */
    :root {
        --ink: #172b4d;
        --muted: #64748b;
        --line: #e4eaf2;
        --surface: #ffffff;
        --surface-soft: #f7f9fc;
        --accent: #287d82;
        --accent-dark: #1f6268;
        --accent-soft: #e8f4f3;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 85% 0%, rgba(215, 234, 244, 0.42), transparent 34%),
            linear-gradient(180deg, #f4f7fb 0%, #fbfcfe 46%, #f6f8fb 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(248, 250, 253, 0.82);
        backdrop-filter: blur(10px);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2.25rem;
        padding-bottom: 4rem;
    }

    /* 제목과 본문 위계 */
    h1, h2, h3 {
        color: var(--ink) !important;
        letter-spacing: -0.035em;
    }

    h1 {
        font-size: clamp(2rem, 3.2vw, 2.7rem) !important;
        font-weight: 820 !important;
        line-height: 1.2 !important;
        margin-bottom: 0.35rem !important;
    }

    h2 {
        font-size: 1.55rem !important;
        font-weight: 760 !important;
        margin-top: 1.25rem !important;
        margin-bottom: 0.55rem !important;
    }

    h3 {
        font-size: 1.12rem !important;
        font-weight: 720 !important;
        line-height: 1.45 !important;
    }

    p, li, label, [data-testid="stMarkdownContainer"] {
        line-height: 1.65;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
        font-size: 0.83rem;
        line-height: 1.55;
    }

    /* 좌측 메뉴 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #edf3f8 0%, #f4f7fb 100%);
        border-right: 1px solid #e1e8f0;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.35rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #52657d;
        font-size: 0.82rem;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] > label {
        color: #31445f;
        font-weight: 700;
    }

    /* 검색 결과 및 저장 모델 카드 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid var(--line) !important;
        border-radius: 18px !important;
        box-shadow: 0 5px 18px rgba(31, 54, 82, 0.045);
        padding: 0.35rem 0.45rem;
        transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #cad8e5 !important;
        box-shadow: 0 9px 24px rgba(31, 54, 82, 0.075);
    }

    /* 입력 영역 */
    [data-testid="stForm"] {
        background: rgba(247, 249, 252, 0.88);
        border: 1px solid #e8edf3;
        border-radius: 15px;
        padding: 1rem 1.05rem 0.75rem;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label {
        color: #344761 !important;
        font-size: 0.9rem !important;
        font-weight: 680 !important;
    }

    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-baseweb="select"] > div {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 10px;
        transition: border-color 150ms ease, box-shadow 150ms ease;
    }

    [data-baseweb="input"] > div:focus-within,
    [data-baseweb="textarea"] > div:focus-within,
    [data-baseweb="select"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(40, 125, 130, 0.12);
    }

    input, textarea {
        color: #1f334d !important;
    }

    /* 버튼 */
    .stButton button,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stLinkButton"] a {
        min-height: 2.55rem;
        border-radius: 10px;
        font-weight: 700;
        letter-spacing: -0.01em;
        transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
    }

    .stButton button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
        background: var(--accent) !important;
        border: 1px solid var(--accent) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 10px rgba(40, 125, 130, 0.16);
    }

    .stButton button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover {
        background: var(--accent-dark) !important;
        border-color: var(--accent-dark) !important;
        box-shadow: 0 6px 14px rgba(40, 125, 130, 0.22);
        transform: translateY(-1px);
    }

    .stButton button[kind="secondary"],
    [data-testid="stFormSubmitButton"] button[kind="secondary"] {
        background: #ffffff !important;
        color: #314761 !important;
        border: 1px solid #d8e1eb !important;
    }

    [data-testid="stLinkButton"] a {
        background: var(--accent-soft);
        border: 1px solid #cfe5e3;
        color: #205e63 !important;
        text-decoration: none !important;
        padding: 0.45rem 0.85rem;
    }

    [data-testid="stLinkButton"] a:hover {
        background: #dcefed;
        border-color: #afd4d1;
        transform: translateY(-1px);
    }

    /* 저장 현황 지표 */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 1rem 1.15rem;
        box-shadow: 0 4px 14px rgba(31, 54, 82, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #61738a;
        font-size: 0.84rem;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 800;
        letter-spacing: -0.035em;
    }

    /* 안내·오류 메시지 */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid #e2e9f0;
        box-shadow: 0 3px 10px rgba(31, 54, 82, 0.035);
    }

    hr {
        border-color: #dfe7ef;
        margin-top: 1.15rem;
        margin-bottom: 1.15rem;
    }

    /* 작은 화면에서 여백과 카드 간격 조정 */
    @media (max-width: 760px) {
        .block-container {
            padding: 1.35rem 1rem 2.5rem;
        }

        h1 {
            font-size: 1.85rem !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px !important;
            padding: 0.2rem 0.25rem;
        }

        [data-testid="stForm"] {
            padding: 0.8rem 0.75rem 0.6rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    "글 생성": "text-generation",
    "언어 번역": "translation",
    "개체명 인식·토큰 분류": "token-classification",
    "빈칸 채우기": "fill-mask",
    "문장 유사도 비교": "sentence-similarity",
    "텍스트 임베딩·특징 추출": "feature-extraction",
    "이미지 분류": "image-classification",
    "이미지 생성": "text-to-image",
    "이미지 설명 생성": "image-to-text",
    "객체 탐지": "object-detection",
    "이미지·텍스트 이해": "image-text-to-text",
    "이미지 기반 질문 답변": "visual-question-answering",
    "이미지 깊이 추정": "depth-estimation",
    "음성 인식·받아쓰기": "automatic-speech-recognition",
    "음성·오디오 분류": "audio-classification",
    "텍스트 음성 변환": "text-to-speech",
    "음악·오디오 생성": "text-to-audio",
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
            "한국어 작업명 예시: 텍스트 분류, 번역, 개체명 인식, 문장 요약, "
            "이미지 분류, 객체 탐지, 이미지 설명, 음성 인식, 음성 생성 등"
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
                st.markdown("**모델 간단 설명**")
                st.write(model.get("description") or api.TASK_DESCRIPTIONS.get(
                    model["task"], "모델의 상세 설명은 모델 페이지에서 확인할 수 있습니다."
                ))
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
