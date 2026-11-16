import re

import requests

API_URL = "https://huggingface.co/api/models"

# 한국어로 입력한 작업 분류를 Hugging Face의 pipeline_tag로 변환합니다.
# 공백, 하이픈, 밑줄 등은 무시하므로 '텍스트 분류'와 '텍스트분류'를
# 같은 검색어로 인식합니다.
TASK_KEYWORD_MAP = {
    # 텍스트 분류 / 감정 분석
    "텍스트분류": "text-classification",
    "문장분류": "text-classification",
    "감정분석": "text-classification",
    "감정분류": "text-classification",
    "문장분류감정분석": "text-classification",
    "textclassification": "text-classification",
    "sentimentanalysis": "text-classification",

    # 문장 / 텍스트 요약
    "문장요약": "summarization",
    "텍스트요약": "summarization",
    "요약": "summarization",
    "summarization": "summarization",

    # 질문 답변
    "질문답변": "question-answering",
    "질의응답": "question-answering",
    "questionanswering": "question-answering",

    # 이미지 분류
    "이미지분류": "image-classification",
    "사진분류": "image-classification",
    "이미지인식": "image-classification",
    "imageclassification": "image-classification",

    # 이미지 생성
    "이미지생성": "text-to-image",
    "그림생성": "text-to-image",
    "texttoimage": "text-to-image",

    # 음성 인식
    "음성인식": "automatic-speech-recognition",
    "음성텍스트변환": "automatic-speech-recognition",
    "automatic speech recognition": "automatic-speech-recognition",
    "automaticspeechrecognition": "automatic-speech-recognition",

    # 글 생성
    "글생성": "text-generation",
    "텍스트생성": "text-generation",
    "문장생성": "text-generation",
    "textgeneration": "text-generation",
}

TASK_DISPLAY_NAMES = {
    "text-classification": "텍스트 분류·감정 분석",
    "summarization": "문장 요약",
    "question-answering": "질문 답변",
    "image-classification": "이미지 분류",
    "text-to-image": "이미지 생성",
    "automatic-speech-recognition": "음성 인식",
    "text-generation": "글 생성",
}

# 검색어 뒤에 아래 표현이 붙어도 작업 분류로 인식합니다.
TASK_SUFFIXES = ("모델", "작업", "검색", "모델검색", "모델추천")


def _normalize_keyword(keyword):
    """한글/영문 작업 분류 검색어의 공백과 구분 기호를 정리합니다."""
    keyword = keyword.casefold().strip()
    return re.sub(r"[\s_\-./·,()]+", "", keyword)


def detect_task_keyword(keyword):
    """한국어 작업 분류 검색어를 Hugging Face pipeline_tag로 변환합니다.

    예: '텍스트 분류', '텍스트분류', '텍스트분류 모델'
        -> 'text-classification'
    일치하는 작업 분류가 없으면 None을 반환합니다.
    """
    normalized = _normalize_keyword(keyword)
    if not normalized:
        return None

    # 가장 긴 별칭부터 확인해 더 구체적인 표현을 우선 인식합니다.
    aliases = sorted(TASK_KEYWORD_MAP.items(), key=lambda item: len(item[0]), reverse=True)

    for alias, task in aliases:
        if normalized == alias:
            return task

        if any(normalized == alias + suffix for suffix in TASK_SUFFIXES):
            return task

    return None


def get_task_display_name(task):
    """API의 영문 작업 분류를 화면에 표시할 한국어 이름으로 변환합니다."""
    return TASK_DISPLAY_NAMES.get(task, task or "분류 정보 없음")


def search_external_models(keyword, task=None, limit=10):
    """Hugging Face에서 모델을 검색하고 필요한 정보를 반환합니다.

    한국어 작업 분류(예: '텍스트분류')가 검색어로 들어오면
    pipeline_tag로 변환하여 검색합니다.
    """
    keyword = (keyword or "").strip()

    # 사용자가 검색창에 '텍스트분류'처럼 한국어 작업명을 입력한 경우 처리합니다.
    detected_task = detect_task_keyword(keyword)
    if detected_task:
        # 목적 선택 메뉴에서 이미 작업을 선택했다면 그 선택을 유지합니다.
        if task is None:
            task = detected_task
        # 한국어 작업명은 모델 ID의 일부가 아니므로 free-text 검색어로 보내지 않습니다.
        keyword = ""

    if keyword == "" and not task:
        return []

    params = {
        "sort": "downloads",
        "direction": -1,
        "limit": limit,
    }

    if keyword:
        params["search"] = keyword

    if task:
        params["pipeline_tag"] = task

    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise RuntimeError("API 응답 형식이 올바르지 않습니다.")

    except requests.RequestException as error:
        raise RuntimeError("네트워크 오류가 발생했습니다. 인터넷 연결을 확인해주세요.") from error
    except ValueError as error:
        raise RuntimeError("응답 데이터를 처리할 수 없습니다.") from error

    results = []

    for item in data:
        model_id = item.get("id", "")
        if not model_id:
            continue

        model_task = item.get("pipeline_tag") or task or "분류 정보 없음"

        model = {
            "model_id": model_id,
            "model_name": model_id,
            "task": model_task,
            "downloads": item.get("downloads") or 0,
            "model_url": "https://huggingface.co/" + model_id,
        }
        results.append(model)

    return results
