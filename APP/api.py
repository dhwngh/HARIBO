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


# 모델 카드에서 '실제 기능'이 아닌 설치·호출·체크포인트 안내를 걸러냅니다.
# 이런 문장은 번역해도 사용자에게 모델의 기능을 설명하지 못하므로 표시하지 않습니다.
TECHNICAL_OR_USAGE_PATTERNS = [
    r"\bfrom_pretrained\b", r"\bAutoModel\b", r"\bAutoTokenizer\b",
    r"\bBARTForConditionalGeneration\b", r"\bAutoModelFor\w+\b",
    r"\b(load|loaded|loading)\s+(this\s+)?(model|checkpoint|weights)\b",
    r"\bshould\s+be\s+loaded\s+into\b", r"\bcheckpoint\b",
    r"\b(pip\s+install|install\s+the\s+package|installation)\b",
    r"\b(usage|how\s+to\s+use|getting\s+started|inference\s+code|example\s+code)\b",
    r"\b(refer\s+to|see)\s+.{0,50}\b(documentation|docs|tutorial)\b",
    r"\b(license|citation|bibliography|training\s+arguments|fine[- ]?tuning\s+instructions)\b",
    r"이\s*체크포인트", r"체크포인트.*(로드|불러와)", r"from_pretrained",
    r"(모델|체크포인트)를?\s*(로드|불러와야)", r"(설치|실행)\s*(방법|명령어|코드)",
    r"(사용|추론)\s*예제", r"코드\s*예제", r"문서(를|에서)\s*(참조|확인)",
    r"라이선스", r"인용\s*방법", r"학습\s*명령어",
]

TECHNICAL_OR_USAGE_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in TECHNICAL_OR_USAGE_PATTERNS),
    re.IGNORECASE,
)

FUNCTIONAL_CUES_RE = re.compile(
    r"\b(designed to|built to|can be used to|used for|capable of|enables?|"
    r"generates?|summarizes?|translates?|classifies?|detects?|recognizes?|"
    r"extracts?|converts?|answers?|identifies?|predicts?|understands?)\b|"
    r"(분류|요약|번역|생성|탐지|인식|추출|변환|예측|답변|질문에 답|설명|분석)"
)

IGNORED_MODEL_CARD_HEADINGS_RE = re.compile(
    r"^(usage|how to use|getting started|installation|inference|examples?|"
    r"training|fine[- ]?tuning|limitations?|license|citation|references?|"
    r"technical details|implementation|requirements|사용 방법|설치|실행 방법|"
    r"추론 방법|학습 방법|코드 예제|라이선스|인용|제한 사항)\b",
    re.IGNORECASE,
)

PREFERRED_MODEL_CARD_HEADINGS_RE = re.compile(
    r"^(overview|model description|description|abstract|intended use|"
    r"what it does|capabilities|summary|개요|모델 설명|모델 소개|주요 기능|활용 목적)\b",
    re.IGNORECASE,
)


def _is_technical_or_usage_text(text: str) -> bool:
    """모델 기능이 아니라 코드 사용법·체크포인트 안내인 문장을 판별합니다."""
    return bool(TECHNICAL_OR_USAGE_RE.search(text or ""))


def _clean_markdown_paragraph(paragraph: str) -> str:
    """Markdown 문단에서 이미지·코드·서식 기호를 제거합니다."""
    lines = []
    in_code_block = False
    for raw_line in paragraph.splitlines():
        line = raw_line.strip()
        if line.startswith("```") or line.startswith("~~~"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not line:
            continue
        if line.startswith(("![", "<", "|")):
            continue
        line = re.sub(r"!\[[^]]*\]\([^)]*\)", " ", line)
        line = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"<[^>]+>", " ", line)
        line = re.sub(r"^[#>*+\-\s]+", "", line)
        line = re.sub(r"[*_~]", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return " ".join(lines).strip()


@lru_cache(maxsize=256)
def _read_model_card_summary(model_id: str) -> str:
    """모델 카드에서 실제 기능을 설명하는 문장을 우선 추출합니다.

    from_pretrained, checkpoint, 설치·사용 예제 같은 기술 안내는 제외합니다.
    기능 소개를 찾지 못하면 빈 문자열을 반환해 작업 분류별 설명을 사용합니다.
    """
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    try:
        response = requests.get(
            url, timeout=(2, 5), headers={"User-Agent": "AI-Model-Finder/1.0"}
        )
        if response.status_code != 200:
            return ""

        raw = response.text[:50000]
        # YAML front matter는 모델의 소개 문장으로 사용하지 않습니다.
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            raw = parts[2] if len(parts) == 3 else raw

        paragraphs = re.split(r"\n\s*\n", raw)
        candidates: list[tuple[int, str]] = []
        current_heading = ""
        ignored_section = False
        preferred_section = False

        for paragraph in paragraphs:
            stripped = paragraph.strip()
            if not stripped:
                continue

            # 제목을 기준으로 Usage/Training 섹션을 건너뛰고 기능 소개 섹션을 우선합니다.
            heading_match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", stripped)
            if heading_match:
                current_heading = re.sub(r"[*`]", "", heading_match.group(1)).strip()
                ignored_section = bool(IGNORED_MODEL_CARD_HEADINGS_RE.search(current_heading))
                preferred_section = bool(PREFERRED_MODEL_CARD_HEADINGS_RE.search(current_heading))
                continue
            if ignored_section:
                continue

            cleaned = _clean_markdown_paragraph(paragraph)
            if len(cleaned) < 35 or _is_technical_or_usage_text(cleaned):
                continue

            # 문단 안에서도 기능을 설명하는 문장만 남깁니다.
            sentences = re.split(r"(?<=[.!?])\s+", cleaned)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 30 or _is_technical_or_usage_text(sentence):
                    continue
                if not re.search(r"[A-Za-z가-힣]", sentence):
                    continue
                score = 3 if preferred_section else 0
                if FUNCTIONAL_CUES_RE.search(sentence):
                    score += 3
                if re.search(r"\b(model|system|tool)\b|모델|시스템", sentence, re.I):
                    score += 1
                # 짧은 소개 문장에 우선순위를 두고, 지나치게 긴 설명은 감점합니다.
                if len(sentence) <= 260:
                    score += 1
                else:
                    score -= 1
                candidates.append((score, sentence))

        if not candidates:
            return ""

        # 기능 설명 신호가 있는 문장을 가장 먼저 선택합니다.
        candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
        best_score, best_sentence = candidates[0]
        if best_score < 2:
            return ""
        best_sentence = re.sub(r"\s+", " ", best_sentence).strip()
        return best_sentence[:500].rstrip() + ("…" if len(best_sentence) > 500 else "")
    except requests.RequestException:
        return ""


@lru_cache(maxsize=512)
def _translate_description_to_korean(text: str) -> str:
    """모델 카드의 기능 설명을 한국어로 번역합니다."""
    text = " ".join((text or "").split()).strip()
    if not text or _is_technical_or_usage_text(text):
        return ""

    hangul_count = len(re.findall(r"[가-힣]", text))
    if hangul_count / max(len(text), 1) >= 0.45:
        return text

    try:
        response = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "auto", "tl": "ko", "dt": "t", "q": text[:1200]},
            timeout=(3, 6),
            headers={"User-Agent": "AI-Model-Finder/1.0"},
        )
        response.raise_for_status()
        data = response.json()
        translated_parts = data[0] if isinstance(data, list) and data else []
        translated = "".join(
            str(part[0]) for part in translated_parts
            if isinstance(part, list) and part and part[0]
        ).strip()
        translated = re.sub(r"\s+", " ", translated)
        translated_hangul_count = len(re.findall(r"[가-힣]", translated))
        if translated and translated_hangul_count >= 3 and not _is_technical_or_usage_text(translated):
            return translated
    except (requests.RequestException, ValueError, TypeError, IndexError, KeyError):
        pass
    return ""


def get_model_description(model_id: str, task: str) -> str:
    """모델의 실제 기능을 설명하고, 기술 안내는 작업 기능 설명으로 대체합니다."""
    summary = _read_model_card_summary(model_id)
    if summary:
        korean_summary = _translate_description_to_korean(summary)
        if korean_summary and not _is_technical_or_usage_text(korean_summary):
            return korean_summary

    # 체크포인트 로딩법이나 라이브러리 안내 대신, 해당 작업에서 모델이 하는 일을 설명합니다.
    return TASK_DESCRIPTIONS.get(
        task,
        "이 모델은 입력 데이터를 분석해 학습된 작업을 수행하는 인공지능 모델입니다. 구체적인 기능은 모델 페이지의 소개를 확인해주세요.",
    )


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

    # 모델 카드 소개를 가져와 한국어로 번역합니다.
    # 카드가 없거나 번역에 실패하면 작업별 한국어 설명을 그대로 사용합니다.
    with ThreadPoolExecutor(max_workers=5) as executor:
        descriptions = list(executor.map(
            lambda row: get_model_description(row["model_id"], row["task"]), results
        ))
    for row, description in zip(results, descriptions):
        row["description"] = description

    return results
