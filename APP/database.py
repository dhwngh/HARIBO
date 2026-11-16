"""SQLite 모델 정보 저장·조회·수정·삭제 모듈."""

import sqlite3
from pathlib import Path

# 실행 위치가 바뀌어도 DB가 항상 APP 폴더 안에 생성되도록 절대 경로를 사용합니다.
DB_NAME = str(Path(__file__).resolve().parent / "models.db")
VALID_PRIORITIES = {"높음", "중", "낮음"}


def connect_db():
    """SQLite 연결을 생성합니다."""
    return sqlite3.connect(DB_NAME, timeout=10)


def create_table():
    """프로그램 실행 시 필요한 테이블과 검색 인덱스를 생성합니다."""
    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS models(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT UNIQUE,
            model_name TEXT NOT NULL,
            task TEXT NOT NULL,
            downloads INTEGER DEFAULT 0,
            use_idea TEXT NOT NULL,
            priority TEXT DEFAULT '중',
            model_url TEXT DEFAULT '',
            source TEXT DEFAULT '직접 등록'
        )""")
        cursor.execute("""CREATE INDEX IF NOT EXISTS idx_models_model_name
                          ON models(model_name)""")
        conn.commit()
    finally:
        conn.close()


def add_model(model_id, model_name, task, downloads, use_idea,
              priority="중", model_url="", source="직접 등록"):
    """모델을 등록합니다. 중복 model_id이면 0, 성공하면 1을 반환합니다."""
    model_id = str(model_id or "").strip()
    model_name = str(model_name or "").strip()
    task = str(task or "").strip()
    use_idea = str(use_idea or "").strip()
    priority = str(priority or "중").strip()
    model_url = str(model_url or "").strip()
    source = str(source or "직접 등록").strip()

    if not model_id or not model_name or not task or not use_idea:
        raise ValueError("모델 ID, 모델명, 작업 분류, 적용 아이디어를 입력해야 합니다.")
    if priority not in VALID_PRIORITIES:
        raise ValueError("우선순위는 높음, 중, 낮음 중에서 선택해야 합니다.")
    try:
        downloads = max(0, int(downloads))
    except (TypeError, ValueError):
        downloads = 0

    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""INSERT INTO models
            (model_id, model_name, task, downloads, use_idea,
             priority, model_url, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_id, model_name, task, downloads, use_idea,
              priority, model_url, source))
        conn.commit()
        return cursor.rowcount
    except sqlite3.IntegrityError:
        conn.rollback()
        return 0
    finally:
        conn.close()


def get_all_models():
    """저장된 전체 모델 목록을 최신 등록순으로 반환합니다."""
    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""SELECT id, model_id, model_name, task, downloads,
            use_idea, priority, model_url, source
            FROM models
            ORDER BY id DESC
        """)
        return cursor.fetchall()
    finally:
        conn.close()


def search_models(keyword=""):
    """모델명에 검색어가 포함된 데이터를 반환합니다. 빈 검색어는 전체 조회합니다."""
    keyword = (keyword or "").strip()
    if not keyword:
        return get_all_models()

    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("""SELECT id, model_id, model_name, task, downloads,
            use_idea, priority, model_url, source
            FROM models
            WHERE model_name LIKE ?
            ORDER BY id DESC
        """, ("%" + keyword + "%",))
        return cursor.fetchall()
    finally:
        conn.close()


def update_priority(model_number, priority):
    """저장된 모델의 우선순위를 변경합니다."""
    if priority not in VALID_PRIORITIES:
        raise ValueError("올바른 우선순위를 선택해주세요.")

    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE models SET priority = ? WHERE id = ?", (priority, model_number))
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def delete_model(model_number):
    """모델을 삭제합니다."""
    conn = connect_db()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM models WHERE id = ?", (model_number,))
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
