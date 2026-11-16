"""Hugging Face 모델 검색 및 한국어 작업 분류 변환 모듈."""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from functools import lru_cache

import requests

API_URL = "https://huggingface.co/api/models"

# 키는 _normalize_keyword()를 거친 형태입니다.
# 사용자가 입력한 한국어 작업명을 Hugging Face의 pipeline_tag로 변환합니다.
TASK_KEYWORD_MAP = {
    # 텍스트 분류 및 감정 분석
    "텍스트분류": "text-classification", "문장분류": "text-classification",
    "감정분석": "text-classification", "감정분류": "text-classification",
    "sentimentanalysis": "text-classification", "textclassification": "text-classification",
    # 요약·질의응답·생성
    "문장요약": "summarization", "텍스트요약": "summarization", "요약": "summarization",
    "summarization": "summarization",
    "질문답변": "question-answering", "질의응답": "question-answering",
    "questionanswering": "question-answering",
    "글생성": "text-generation", "텍스트생성": "text-generation", "문장생성": "text-generation",
    "textgeneration": "text-generation",
    # 번역·언어 분석
    "번역": "translation", "기계번역": "translation", "언어번역": "translation",
    "translation": "translation",
    "개체명인식": "token-classification", "고유명사인식": "token-classification",
    "토큰분류": "token-classification", "tokenclassification": "token-classification",
    "빈칸채우기": "fill-mask", "마스크언어모델": "fill-mask", "fillmask": "fill-mask",
    "문장유사도": "sentence-similarity", "문장비교": "sentence-similarity",
    "sentence similarity": "sentence-similarity",
    "문장임베딩": "feature-extraction", "텍스트임베딩": "feature-extraction",
    "특징추출": "feature-extraction", "featureextraction": "feature-extraction",
    # 이미지
    "이미지분류": "image-classification", "사진분류": "image-classification",
    "이미지인식": "image-classification", "imageclassification": "image-classification",
    "이미지생성": "text-to-image", "그림생성": "text-to-image", "texttoimage": "text-to-image",
    "이미지설명": "image-to-text", "이미지캡션": "image-to-text", "사진설명": "image-to-text",
    "imagetotext": "image-to-text",
    "객체탐지": "object-detection", "물체탐지": "object-detection", "물체인식": "object-detection",
    "objectdetection": "object-detection",
    "이미지검색": "image-text-to-text", "이미지질의": "image-text-to-text",
    "시각질의응답": "visual-question-answering", "이미지질문답변": "visual-question-answering",
    "visualquestionanswering": "visual-question-answering",
    "이미지깊이추정": "depth-estimation", "깊이추정": "depth-estimation",
    # 음성·오디오
    "음성인식": "automatic-speech-recognition", "음성텍스트변환": "automatic-speech-recognition",
    "받아쓰기": "automatic-speech-recognition", "automaticspeechrecognition": "automatic-speech-recognition",
    "음성분류": "audio-classification", "오디오분류": "audio-classification",
    "audioclassification": "audio-classification",
    "음성합성": "text-to-speech", "텍스트음성변환": "text-to-speech", "음성생성": "text-to-speech",
    "texttospeech": "text-to-speech",
    "음악생성": "text-to-audio", "오디오생성": "text-to-audio", "texttoaudio": "text-to-audio",
}

TASK_DISPLAY_NAMES = {
    "text-classification": "텍스트 분류·감정 분석",
    "summarization": "문장 요약",
    "question-answering": "질문 답변",
    "text-generation": "글 생성",
    "translation": "언어 번역",
    "token-classification": "개체명 인식·토큰 분류",
    "fill-mask": "빈칸 채우기",
    "sentence-similarity": "문장 유사도 비교",
    "feature-extraction": "텍스트 임베딩·특징 추출",
    "image-classification": "이미지 분류",
    "text-to-image": "이미지 생성",
    "image-to-text": "이미지 설명 생성",
    "object-detection": "객체 탐지",
    "image-text-to-text": "이미지·텍스트 이해",
    "visual-question-answering": "이미지 기반 질문 답변",
    "depth-estimation": "이미지 깊이 추정",
    "automatic-speech-recognition": "음성 인식·받아쓰기",
    "audio-classification": "음성·오디오 분류",
    "text-to-speech": "텍스트 음성 변환",
    "text-to-audio": "음악·오디오 생성",
}

TASK_DESCRIPTIONS = {
    "text-classification": "문장이나 문서의 주제·감정·의도를 분류합니다.",
    "summarization": "긴 글의 핵심 내용을 짧고 간결하게 요약합니다.",
    "question-answering": "주어진 문맥이나 지식을 바탕으로 질문에 답합니다.",
    "text-generation": "입력한 문장이나 지시를 바탕으로 새로운 글을 생성합니다.",
    "translation": "한 언어로 작성된 문장을 다른 언어로 번역합니다.",
    "token-classification": "문장에서 사람·장소·기관 등의 개체명이나 토큰 단위를 찾아 분류합니다.",
    "fill-mask": "문장의 가려진 단어를 문맥에 맞게 예측합니다.",
    "sentence-similarity": "두 문장의 의미가 얼마나 비슷한지 비교합니다.",
    "feature-extraction": "텍스트를 숫자 벡터로 바꾸어 검색·분류 등에 활용할 수 있게 합니다.",
    "image-classification": "이미지 전체가 어떤 사물이나 범주에 해당하는지 분류합니다.",
    "text-to-image": "텍스트 설명을 바탕으로 새로운 이미지를 생성합니다.",
    "image-to-text": "이미지에 포함된 장면이나 사물을 문장으로 설명합니다.",
    "object-detection": "이미지 안의 여러 객체 위치를 상자 형태로 찾고 종류를 분류합니다.",
    "image-text-to-text": "이미지와 텍스트를 함께 입력받아 내용을 이해하고 응답합니다.",
    "visual-question-answering": "이미지를 보고 사용자가 입력한 질문에 답합니다.",
    "depth-estimation": "이미지의 각 영역이 카메라로부터 얼마나 떨어져 있는지 추정합니다.",
    "automatic-speech-recognition": "음성·녹음 파일을 인식하여 텍스트로 변환합니다.",
    "audio-classification": "소리의 종류나 음성의 특징을 분석하여 범주를 분류합니다.",
    "text-to-speech": "입력한 텍스트를 사람이 들을 수 있는 음성으로 변환합니다.",
    "text-to-audio": "텍스트 설명을 바탕으로 음악이나 효과음 등 오디오를 생성합니다.",
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


@lru_cache(maxsize=256)
def _read_model_card_summary(model_id: str) -> str:
    """Hugging Face 모델 카드에서 소개 문단을 짧게 가져옵니다."""
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    try:
        response = requests.get(url, timeout=(2, 4), headers={"User-Agent": "AI-Model-Finder/1.0"})
        if response.status_code != 200:
            return ""
        text = response.text[:30000]
        # YAML front matter는 설명 문장으로 취급하지 않습니다.
        if text.startswith("---"):
            parts = text.split("---", 2)
            text = parts[2] if len(parts) == 3 else text
        paragraphs = re.split(r"\n\s*\n", text)
        for paragraph in paragraphs:
            lines = []
            for line in paragraph.splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "<", "```", "![", "|", "- ", "* ")):
                    continue
                line = re.sub(r"`([^`]+)`", r"\1", line)
                line = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", line)
                line = re.sub(r"[*_>#]", "", line).strip()
                if line:
                    lines.append(line)
            summary = " ".join(lines)
            if len(summary) >= 35:
                return summary[:260].rstrip() + ("…" if len(summary) > 260 else "")
    except requests.RequestException:
        pass
    return ""


def get_model_description(model_id: str, task: str) -> str:
    """모델 카드 소개가 있으면 우선 사용하고, 없으면 작업 설명을 제공합니다."""
    summary = _read_model_card_summary(model_id)
    if summary:
        return summary
    return TASK_DESCRIPTIONS.get(task, "이 모델의 자세한 설명은 모델 페이지에서 확인할 수 있습니다.")


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
            "description": TASK_DESCRIPTIONS.get(model_task, "모델의 상세 설명은 모델 페이지에서 확인할 수 있습니다."),
        })

    # 모델 카드 소개 문단을 병렬로 가져와 검색 결과에 보충합니다.
    with ThreadPoolExecutor(max_workers=5) as executor:
        descriptions = list(executor.map(
            lambda row: _read_model_card_summary(row["model_id"]), results
        ))
    for row, description in zip(results, descriptions):
        if description:
            row["description"] = description

    return results
