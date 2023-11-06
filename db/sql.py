import sqlite3

# 데이터베이스 파일 경로
DATABASE_PATH = 'conversation.db'

def get_db_connection():
    """데이터베이스 연결을 반환합니다."""
    conn = sqlite3.connect(DATABASE_PATH)
    return conn

def create_table():
    """대화 내용을 저장할 테이블을 생성합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_conversation(user_input, bot_response):
    """대화 내용을 데이터베이스에 저장합니다."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversations (user_input, bot_response)
        VALUES (?, ?)
    ''', (user_input, bot_response))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_table()