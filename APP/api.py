"""Hugging Face 모델 검색 및 한국어 작업 분류 변환 모듈."""

import re
from typing import Any

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        if line.startswith(("#", "![", "<", "|")):
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
    """모델 카드에서 모델 고유의 목적·전문 분야·학습 특징을 추출합니다.

    코드 실행법, 체크포인트 로딩, 설치 안내는 제외합니다. 모델 카드의
    소개 영역과 기능/활용 섹션을 우선 살펴보고, 모델마다 다른 문장을 반환합니다.
    """
    if not model_id or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+", model_id):
        return ""

    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    try:
        response = requests.get(
            url, timeout=(2, 7), headers={"User-Agent": "AI-Model-Finder/1.0"}
        )
        if response.status_code != 200:
            return ""

        raw = response.text[:80000]
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            raw = parts[2] if len(parts) == 3 else raw

        # 모델별 설명이 담길 가능성이 높은 섹션을 우선합니다.
        preferred_sections = re.compile(
            r"^(overview|model description|description|abstract|intended use|"
            r"what it does|capabilities|model overview|uses|use cases|개요|"
            r"모델 설명|모델 소개|주요 기능|활용 목적|사용 목적)\b", re.I
        )
        ignored_sections = re.compile(
            r"^(usage|how to use|getting started|installation|inference|examples?|"
            r"training|fine[- ]?tuning|limitations?|license|citation|references?|"
            r"technical details|implementation|requirements|倫理|사용 방법|설치|"
            r"실행 방법|추론 방법|학습 방법|코드 예제|라이선스|인용|제한 사항)\b", re.I
        )

        candidates: list[tuple[int, int, str]] = []
        current_heading = ""
        ignored_section = False
        preferred_section = False
        paragraph_index = 0

        for paragraph in re.split(r"\n\s*\n", raw):
            paragraph_index += 1
            stripped = paragraph.strip()
            if not stripped:
                continue

            heading_match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", stripped)
            if heading_match:
                current_heading = re.sub(r"[*`]", "", heading_match.group(1)).strip()
                ignored_section = bool(ignored_sections.search(current_heading))
                preferred_section = bool(preferred_sections.search(current_heading))
                continue
            if ignored_section:
                continue

            cleaned = _clean_markdown_paragraph(paragraph)
            if len(cleaned) < 25 or _is_technical_or_usage_text(cleaned):
                continue

            for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
                sentence = sentence.strip(" -•\t")
                if len(sentence) < 25 or len(sentence) > 700:
                    continue
                if _is_technical_or_usage_text(sentence):
                    continue
                # 모델 카드에서 흔히 반복되는 범용 템플릿·다운로드 안내를 제외합니다.
                if re.search(
                    r"(download( the)? model|click here|available on hugging face|"
                    r"this repository contains|for more information|see the documentation|"
                    r"모델을 다운로드|자세한 내용은.*문서|이 저장소에는)", sentence, re.I
                ):
                    continue

                score = 0
                if preferred_section:
                    score += 7
                if FUNCTIONAL_CUES_RE.search(sentence):
                    score += 3
                if re.search(
                    r"(fine[- ]?tun|trained|pretrained|speciali[sz]ed|optim[iz]ed|"
                    r"multilingual|korean|english|medical|clinical|financial|"
                    r"sentiment|emotion|spam|toxicity|zero[- ]shot|instruction|"
                    r"dataset|corpus|domain|benchmark|특화|미세 조정|학습 데이터|"
                    r"다국어|한국어|영어|의료|금융|감정|스팸|유해|대화|코드)",
                    sentence, re.I
                ):
                    score += 3
                if re.search(r"\b(model|system|encoder|decoder|transformer)\b|모델|시스템", sentence, re.I):
                    score += 1
                # 첫 소개 문단을 뒤쪽의 실험·인용 문단보다 우선합니다.
                score += max(0, 3 - paragraph_index // 3)
                if len(sentence) <= 320:
                    score += 1
                candidates.append((score, paragraph_index, sentence))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: (item[0], -item[1], -len(item[2])), reverse=True)
        best_score, _, best_sentence = candidates[0]
        if best_score < 3:
            return ""

        # 두 번째로 높은 문장이 첫 문장과 같은 소개 문단의 보완 설명이면 덧붙입니다.
        selected = [best_sentence]
        for score, _, sentence in candidates[1:]:
            if score < 3 or sentence == best_sentence:
                continue
            if len(" ".join(selected + [sentence])) <= 650:
                selected.append(sentence)
            break
        return re.sub(r"\s+", " ", " ".join(selected)).strip()
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


# 검색 결과의 설명은 모델 카드의 설치·호출 안내를 그대로 보여주지 않고,
# Hugging Face 작업 분류를 바탕으로 아래의 통일된 한국어 문장으로 표시합니다.
TASK_FUNCTION_PHRASES = {
    "text-classification": "텍스트 분류",
    "summarization": "문장 요약",
    "question-answering": "질문에 대한 답변 생성",
    "text-generation": "텍스트 생성",
    "translation": "언어 번역",
    "token-classification": "개체명 및 토큰 분류",
    "fill-mask": "문장의 빈칸에 들어갈 단어 예측",
    "sentence-similarity": "문장 간 의미 유사도 비교",
    "feature-extraction": "텍스트 특징 추출 및 임베딩 생성",
    "image-classification": "이미지 분류",
    "text-to-image": "텍스트를 바탕으로 이미지 생성",
    "image-to-text": "이미지 내용을 텍스트로 설명",
    "object-detection": "이미지 속 객체 탐지",
    "image-text-to-text": "이미지와 텍스트 내용 이해 및 응답",
    "visual-question-answering": "이미지에 관한 질문에 답변",
    "depth-estimation": "이미지의 깊이 추정",
    "automatic-speech-recognition": "음성을 텍스트로 변환",
    "audio-classification": "오디오 분류",
    "text-to-speech": "텍스트를 음성으로 변환",
    "text-to-audio": "텍스트를 바탕으로 오디오 생성",
}

MODEL_DOMAIN_HINTS = {
    "sentiment": "감정",
    "emotion": "세부 감정",
    "spam": "스팸 메시지",
    "toxicity": "유해 표현",
    "hate": "혐오 표현",
    "sarcasm": "반어·비꼼 표현",
    "topic": "문서 주제",
    "news": "뉴스 주제",
    "review": "상품·서비스 후기",
    "ner": "사람·기관·장소 등의 고유명사",
    "medical": "의료 분야 정보",
    "clinical": "임상·의료 분야 정보",
    "finance": "금융 분야 정보",
    "legal": "법률 분야 정보",
}

MODEL_CONTEXT_HINTS = {
    "twitter": "짧은 소셜 미디어 문장",
    "tweet": "짧은 소셜 미디어 문장",
    "korean": "한국어 문장",
    "english": "영어 문장",
    "multilingual": "여러 언어의 문장",
    "multilang": "여러 언어의 문장",
    "medical": "의료 분야 문서",
    "clinical": "임상·의료 분야 문서",
    "finance": "금융 분야 문서",
    "legal": "법률 문서",
    "code": "프로그래밍 코드",
}

MODEL_ARCHITECTURE_HINTS = {
    "distilbert": "DistilBERT 기반의 경량화 구조",
    "tiny": "작은 크기와 빠른 처리에 초점을 둔 초경량 구조",
    "small": "상대적으로 작은 크기와 효율적인 처리에 초점을 둔 소형 구조",
    "large": "대규모 모델 구조",
    "instruction": "사용자 지시 수행에 맞춘 추가 학습 방식",
    "chat": "대화형 응답에 맞춘 추가 학습 방식",
    "zero-shot": "별도의 작업별 추가 학습 없이 다양한 범주에 적용하는 제로샷 방식",
    "zero_shot": "별도의 작업별 추가 학습 없이 다양한 범주에 적용하는 제로샷 방식",
}


def _feature_from_model_metadata(model_id: str, tags: list[str] | None = None) -> dict[str, str]:
    """모델 ID와 Hugging Face 태그에서 확인 가능한 고유 특성을 정리합니다."""
    haystack = (model_id or "").casefold() + " " + " ".join(str(tag) for tag in (tags or [])).casefold()

    def contains_token(token: str) -> bool:
        if token in {"ko", "en", "ja", "zh", "fr", "de", "es"}:
            return bool(re.search(r"(?:^|[-_/:])" + re.escape(token) + r"(?:$|[-_/:])", haystack))
        return token in haystack

    info = {"domain": "", "context": "", "architecture": "", "language": ""}
    for token, value in MODEL_DOMAIN_HINTS.items():
        if contains_token(token):
            info["domain"] = value
            break
    for token, value in MODEL_CONTEXT_HINTS.items():
        if contains_token(token):
            info["context"] = value
            break
    for token, value in MODEL_ARCHITECTURE_HINTS.items():
        if contains_token(token):
            info["architecture"] = value
            break

    language_names = {
        "ko": "한국어", "en": "영어", "ja": "일본어", "zh": "중국어",
        "fr": "프랑스어", "de": "독일어", "es": "스페인어", "multi": "다국어",
    }
    for tag in tags or []:
        tag_text = str(tag).casefold()
        if tag_text.startswith("language:"):
            code = tag_text.split(":", 1)[1]
            language = language_names.get(code)
            if language:
                info["language"] = language
                if not info["context"]:
                    info["context"] = language + " 문장"
                break
    return info


def _object_particle(text: str) -> str:
    """한국어 목적격 조사 을/를을 앞말의 받침에 맞게 선택합니다."""
    if not text:
        return "을"
    last = text[-1]
    if "가" <= last <= "힣":
        jongseong = (ord(last) - 0xAC00) % 28
        return "을" if jongseong else "를"
    return "을"


def _metadata_based_description(model_id: str, task: str, tags: list[str] | None = None) -> str:
    """모델 카드가 없을 때 ID/태그에서 확인된 특징을 자연스러운 문장으로 만듭니다."""
    info = _feature_from_model_metadata(model_id, tags)
    task_name = get_task_display_name(task)

    task_actions = {
        "text-classification": "텍스트의 주제나 범주를 분류",
        "summarization": "긴 글의 핵심 내용을 요약",
        "question-answering": "주어진 질문에 답변",
        "text-generation": "새로운 텍스트를 생성",
        "translation": "문장을 다른 언어로 번역",
        "token-classification": "문장 속 개체명과 토큰을 분류",
        "fill-mask": "문맥에 맞는 빈칸의 단어를 예측",
        "sentence-similarity": "두 문장 사이의 의미 유사도를 비교",
        "feature-extraction": "텍스트의 의미적 특징을 벡터로 추출",
        "image-classification": "이미지에 포함된 주요 대상이나 범주를 분류",
        "text-to-image": "텍스트 설명을 바탕으로 이미지를 생성",
        "image-to-text": "이미지에 담긴 장면과 사물을 문장으로 설명",
        "object-detection": "이미지 속 객체의 종류와 위치를 탐지",
        "image-text-to-text": "이미지와 텍스트 정보를 함께 이해하고 응답",
        "visual-question-answering": "이미지를 바탕으로 사용자의 질문에 답변",
        "depth-estimation": "이미지 속 사물과 배경의 상대적인 깊이를 추정",
        "automatic-speech-recognition": "음성 내용을 텍스트로 변환",
        "audio-classification": "소리의 종류나 음성의 특징을 분류",
        "text-to-speech": "입력한 텍스트를 음성으로 변환",
        "text-to-audio": "텍스트 설명에 맞는 음악이나 효과음을 생성",
    }

    domain_targets = {
        "감정": ("긍정·부정 감정", "분류"),
        "세부 감정": ("기쁨·슬픔·분노 등 세부 감정", "분류"),
        "스팸 메시지": ("스팸 메시지", "탐지"),
        "유해 표현": ("유해하거나 공격적인 표현", "탐지"),
        "혐오 표현": ("혐오·차별 표현", "탐지"),
        "반어·비꼼 표현": ("반어적이거나 비꼬는 표현", "판별"),
        "문서 주제": ("문서의 핵심 주제", "분류"),
        "뉴스 주제": ("뉴스의 주제", "분류"),
        "상품·서비스 후기": ("후기의 내용과 감정", "분석"),
        "사람·기관·장소 등의 고유명사": ("사람·기관·장소 등의 고유명사", "인식"),
        "의료 분야 정보": ("의료 관련 정보", "분류"),
        "임상·의료 분야 정보": ("임상 기록의 주요 정보", "분류"),
        "금융 분야 정보": ("금융 관련 정보", "분류"),
        "법률 분야 정보": ("법률 문서의 주요 정보", "분류"),
    }

    description = ""
    if info["domain"]:
        target, domain_action = domain_targets.get(info["domain"], (info["domain"], "분석"))
        if info["context"]:
            context = info["context"]
            if context.endswith("문장"):
                subject = f"{context}에 담긴 {target}"
            elif context.endswith("문서"):
                subject = f"{context}에서 {target}"
            elif context == "프로그래밍 코드":
                subject = f"{context}에 포함된 {target}"
            else:
                subject = f"{context}의 {target}"
        else:
            subject = target
        particle = _object_particle(target)
        description = f"이 AI 모델은 {subject}{particle} {domain_action}하도록 특화된 모델입니다."
    elif info["context"]:
        context = info["context"]
        if task == "text-classification":
            description = f"이 AI 모델은 {context}의 주제나 범주를 분류하는 모델입니다."
        elif task == "summarization":
            description = f"이 AI 모델은 {context}의 핵심 내용을 요약하는 모델입니다."
        elif task == "translation":
            description = f"이 AI 모델은 {context}를 다른 언어로 번역하는 모델입니다."
        else:
            description = f"이 AI 모델은 {context}를 대상으로 {task_actions.get(task, task_name + ' 기능을 수행')}하는 모델입니다."
    else:
        description = f"이 AI 모델은 {task_actions.get(task, task_name + ' 기능을 수행')}하는 모델입니다."

    if info["architecture"]:
        description = description.rstrip(".") + f". 또한 {info['architecture']}를 사용합니다."
    return description

def _format_model_card_summary(model_id: str, summary: str) -> str:
    """번역된 모델 카드 문장을 화면에서 읽기 쉬운 한국어 한두 문장으로 정리합니다."""
    summary = re.sub(r"\s+", " ", (summary or "")).strip()
    if not summary:
        return ""

    # 번역문이 모델 자체를 주어로 시작하면 '이 AI 모델은' 형식에 맞춰 주어를 정리합니다.
    names = sorted({model_id, model_id.rsplit("/", 1)[-1]}, key=len, reverse=True)
    for name in names:
        if name:
            summary = re.sub(
                r"^" + re.escape(name) + r"\s*(?:은|는|이|가)\s*", "", summary, flags=re.I
            )
    summary = re.sub(r"^(?:이\s*)?(?:AI\s*)?모델은\s*", "", summary, flags=re.I)
    summary = re.sub(r"^(?:이\s*)?모델이\s*", "", summary, flags=re.I)
    # Model Card 번역문에서 BERT, RoBERTa 등 영문 모델명이 주어로 남는 경우 제거합니다.
    summary = re.sub(r"^[A-Z][A-Za-z0-9_.-]{1,40}\s*(?:은|는|이|가)\s*", "", summary)
    summary = summary.strip(" .。;；")
    if not summary:
        return ""

    # 모델 카드의 문장을 그대로 길게 늘어놓지 않고 핵심 특징 위주로 표시합니다.
    if len(summary) > 420:
        summary = summary[:417].rsplit(" ", 1)[0] + "…"

    # 카드 원문이 이미 완결된 서술형 문장이면 문장 끝을 중복하지 않습니다.
    if re.search(r"(?:모델|시스템|도구|모델군)(?:입니다|이다|로 설계되었습니다|로 학습되었습니다)[.!。]?$", summary):
        return "이 AI 모델은 " + summary.rstrip(".。") + "."
    if re.search(r"(?:합니다|됩니다|되었습니다|됐습니다|입니다|있습니다|가능합니다|되어 있습니다)[.!。]?$", summary):
        return "이 AI 모델은 " + summary.rstrip(".。") + "."
    return "이 AI 모델은 " + summary.rstrip(".。") + " 특징을 가진 모델입니다."


def get_model_description(model_id: str, task: str, tags: list[str] | None = None) -> str:
    """모델마다 다른 Model Card 설명을 우선 사용하고, 부족하면 확인 가능한 메타데이터로 보완합니다.

    동일한 pipeline_tag여도 감정 분석, 스팸 탐지, 한국어 특화 등 모델별 차이를
    설명에 반영합니다. 모델 카드가 비어 있으면 추측으로 성능을 단정하지 않습니다.
    """
    # 1) 모델별 Model Card의 실제 소개 문장을 우선합니다.
    model_card_summary = _read_model_card_summary(model_id)
    translated_summary = _translate_description_to_korean(model_card_summary)
    if translated_summary:
        formatted = _format_model_card_summary(model_id, translated_summary)
        if formatted and not _is_technical_or_usage_text(formatted):
            return formatted

    # 2) 카드 설명이 없거나 번역되지 않으면 모델 ID·API 태그에서 확인되는
    # 전문 분야, 언어, 모델 구조 등의 고유 특징을 반영합니다.
    return _metadata_based_description(model_id, task, tags)

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
            "tags": item.get("tags") or [],
            "description": TASK_DESCRIPTIONS.get(model_task, "모델의 상세 설명은 모델 페이지에서 확인할 수 있습니다."),
        })

    # 각 모델 카드에서 고유 특징을 찾고 한국어로 번역합니다.
    # 여러 모델 설명을 병렬로 가져와 검색 화면이 지나치게 늦게 뜨는 것을 줄입니다.
    if results:
        worker_count = min(5, len(results))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_rows = {
                executor.submit(
                    get_model_description,
                    row["model_id"],
                    row["task"],
                    row.get("tags", []),
                ): row
                for row in results
            }
            for future in as_completed(future_rows):
                row = future_rows[future]
                try:
                    row["description"] = future.result()
                except Exception:
                    # 특정 모델 카드의 오류가 전체 검색 결과를 중단시키지 않게 합니다.
                    row["description"] = get_model_description(row["model_id"], row["task"], row.get("tags", []))

    return results
