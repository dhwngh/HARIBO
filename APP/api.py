"""Hugging Face 모델 검색 및 한국어 작업 분류 변환 모듈."""

import re
from typing import Any

import requests

API_URL = "https://huggingface.co/api/models"

# 키는 _normalize_keyword()를 거친 형태입니다.
# 사용자가 입력한 한국어 작업명을 Hugging Face의 pipeline_tag로 변환합니다.
TASK_KEYWORD_MAP = {
    # 텍스트 분류 / 감정 분석
    "텍스트분류": "text-classification",
    "문장분류": "text-classification",
    "감정분석": "text-classification",
    "감정분류": "text-classification",
    "sentimentanalysis": "text-classification",
    "textclassification": "text-classification",

    # 요약
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
TASK_SUFFIXES = ("모델", "작업", "검색", "모델검색", "모델추천", "모델찾기")


def _normalize_keyword(keyword: str) -> str:
    """대소문자, 공백, 하이픈 등 차이를 제거해 비교하기 쉽게 만듭니다."""
    keyword = (keyword or "").casefold().strip()
    return re.sub(r"[\s_\-./·,()]+", "", keyword)


def detect_task_keyword(keyword: str):
    """한국어/영어 작업명 별칭을 Hugging Face pipeline_tag로 변환합니다.

    예: '텍스트 분류', '텍스트분류 모델', 'text-classification'
        -> 'text-classification'

    등록된 별칭이 아니면 None을 반환합니다. 임의의 한국어 문장을
    번역하거나 의미적으로 해석하는 기능은 아닙니다.
    """
    normalized = _normalize_keyword(keyword)
    if not normalized:
        return None

    # 가장 긴 별칭부터 비교해 구체적인 표현을 우선 인식합니다.
    aliases = sorted(TASK_KEYWORD_MAP.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, task in aliases:
        if normalized == alias:
            return task
        if any(normalized == alias + suffix for suffix in TASK_SUFFIXES):
            return task
    return None


def get_task_display_name(task: str) -> str:
    """API 작업 분류 코드를 화면용 한국어 이름으로 바꿉니다."""
    return TASK_DISPLAY_NAMES.get(task, task or "분류 정보 없음")


def _friendly_http_error(status_code: int) -> str:
    """HTTP 상태 코드에 맞는 사용자가 이해하기 쉬운 오류 문구를 반환합니다."""
    if status_code == 429:
        return "검색 요청이 많아 잠시 제한되었습니다. 잠시 후 다시 검색해주세요."
    if status_code in (401, 403):
        return "Hugging Face API 접근이 거부되었습니다. 잠시 후 다시 시도해주세요."
    if 500 <= status_code <= 599:
        return "Hugging Face 서버에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    return "모델 검색 요청에 실패했습니다. 입력한 검색어를 확인한 뒤 다시 시도해주세요."


def search_external_models(keyword: str, task: str = None, limit: int = 10) -> list[dict[str, Any]]:
    """Hugging Face API에서 모델을 검색합니다.

    - keyword: 모델 ID/이름 검색어 또는 등록된 한국어 작업명
    - task: Hugging Face pipeline_tag
    - 한국어 작업명이 입력되면 pipeline_tag 필터로 변환합니다.
    - 다운로드 수가 많은 모델부터 정렬합니다.
    """
    keyword = (keyword or "").strip()
    detected_task = detect_task_keyword(keyword)

    if detected_task:
        # 작업 분류 선택 메뉴에서 값이 지정되지 않은 경우 한국어 검색어를 사용합니다.
        if task is None:
            task = detected_task
        # 작업 분류는 모델 ID 검색어가 아니므로 search 파라미터에는 넣지 않습니다.
        keyword = ""

    if not keyword and not task:
        return []

    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 10

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
        response = requests.get(API_URL, params=params, timeout=(5, 15))
        response.raise_for_status()
        data = response.json()
    except requests.Timeout as error:
        raise RuntimeError("모델 검색 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.") from error
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else 0
        raise RuntimeError(_friendly_http_error(status_code)) from error
    except requests.RequestException as error:
        raise RuntimeError("네트워크 연결에 실패했습니다. 인터넷 연결을 확인한 뒤 다시 시도해주세요.") from error
    except ValueError as error:
        raise RuntimeError("검색 서버의 응답을 읽지 못했습니다. 잠시 후 다시 시도해주세요.") from error

    if not isinstance(data, list):
        raise RuntimeError("검색 서버의 응답 형식이 예상과 다릅니다. 잠시 후 다시 시도해주세요.")

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue

        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue

        raw_downloads = item.get("downloads") or 0
        try:
            downloads = max(0, int(raw_downloads))
        except (TypeError, ValueError):
            downloads = 0

        model_task = item.get("pipeline_tag") or task or "분류 정보 없음"
        results.append({
            "model_id": model_id,
            "model_name": model_id,
            "task": model_task,
            "downloads": downloads,
            "model_url": "https://huggingface.co/" + model_id,
        })

    return results
