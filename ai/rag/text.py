"""키워드 검색용 토큰화와 한자 독음 보강 (기획서 §5, §11)."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import hanja
from kiwipiepy import Kiwi


HANJA_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")

# 명사·어근·외국어·한자·숫자만 색인한다. 동사·형용사는 질문의 '알려줘' 같은 표현이 잡음이 되어 뺀다.
KEEP_TAGS = {"NNG", "NNP", "NNB", "NR", "XR", "SL", "SH", "SN"}


@lru_cache(maxsize=1)
def _kiwi() -> Kiwi:
    return Kiwi()


def has_hanja(text: str) -> bool:
    return bool(HANJA_RE.search(text))


def _read(span: str) -> str:
    # 호환용 한자(令 등)는 사전에 없어서 NFKC로 통합 한자로 바꾼 뒤 읽는다
    return hanja.translate(unicodedata.normalize("NFKC", span), "substitution")


def hanja_reading(text: str) -> str:
    """한자 구간만 한글 독음으로 바꾼 문자열. 예: 白磁大缸 → 백자대항."""
    return HANJA_RE.sub(lambda m: _read(m.group(0)), text)


def with_hanja_reading(text: str) -> str:
    """한자가 있으면 원문 뒤에 독음을 덧붙인다. 검색용 텍스트에만 쓰고 원본 데이터는 바꾸지 않는다."""
    if not has_hanja(text):
        return text
    # 독음은 띄어쓰기가 없으므로(이순신신도비명) 자동 띄어쓰기로 질문과 같은 형태소가 나오게 한다
    readings = [_kiwi().space(_read(span)) for span in HANJA_RE.findall(text)]
    return f"{text}\n(독음: {' '.join(readings)})"


def tokenize(text: str) -> list[str]:
    """BM25용 토큰. 형태소 + 한자는 글자 단위로도 쪼개 부분 일치를 허용한다."""
    tokens: list[str] = []
    for token in _kiwi().tokenize(text):
        if token.tag not in KEEP_TAGS:
            continue
        form = token.form.lower()
        tokens.append(form)
        if token.tag == "SH" and len(form) > 1:
            tokens.extend(form)
    return tokens
