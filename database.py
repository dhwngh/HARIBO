import sqlite3
from pathlib import Path

# Streamlit Community Cloud에서도 앱 파일과 같은 폴더에 DB를 생성합니다.
DB_NAME = str(Path(__file__).resolve().parent / "models.db")


def connect_db():
    return sqlite3.connect(DB_NAME, timeout=10)


def create_table():
    conn = connect_db()
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

    conn.commit()
    conn.close()


def add_model(model_id, model_name, task, downloads, use_idea,
              priority="중", model_url="", source="직접 등록"):
    conn = connect_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""INSERT INTO models
            (model_id, model_name, task, downloads, use_idea,
             priority, model_url, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_id, model_name, task, downloads, use_idea,
              priority, model_url, source))
        conn.commit()
        changed = cursor.rowcount
    except sqlite3.IntegrityError:
        changed = 0
    finally:
        conn.close()

    return changed


def get_all_models():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""SELECT id, model_id, model_name, task, downloads,
        use_idea, priority, model_url, source
        FROM models
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


def search_models(keyword):
    keyword = keyword.strip()

    if keyword == "":
        return get_all_models()

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""SELECT id, model_id, model_name, task, downloads,
        use_idea, priority, model_url, source
        FROM models
        WHERE model_name LIKE ?
        ORDER BY id DESC
    """, ("%" + keyword + "%",))

    rows = cursor.fetchall()
    conn.close()
    return rows


def update_priority(model_number, priority):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""UPDATE models SET priority = ? WHERE id = ?
    """, (priority, model_number))

    conn.commit()
    changed = cursor.rowcount
    conn.close()
    return changed


def delete_model(model_number):
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""DELETE FROM models WHERE id = ?
    """, (model_number,))

    conn.commit()
    changed = cursor.rowcount
    conn.close()
    return changed
