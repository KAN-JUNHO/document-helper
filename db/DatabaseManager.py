import json
import sqlite3


class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS messages
                             (id INTEGER PRIMARY KEY AUTOINCREMENT,
                              session_id TEXT,
                              user_query TEXT,
                              k TEXT,
                              fetch_k TEXT,
                              lambda_mult TEXT,
                              score_threshold TEXT,
                              response TEXT,
                              source_docs TEXT,
                              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
        )
        self.conn.commit()

    def insert_message(
        self,
        user_query,
        k,
        fetch_k,
        lambda_mult,
        score_threshold,
        response,
        source_docs,
        session_id,
    ):
        # Insert the data into the database, using the JSON string for source_docs
        self.cursor.execute(
            """INSERT INTO messages (user_query, k,fetch_k,lambda_mult,score_threshold, response, source_docs, session_id)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_query,
                k,
                fetch_k,
                lambda_mult,
                score_threshold,
                response,
                source_docs,
                session_id,
            ),
        )
        self.conn.commit()

    def fetch_all_messages(self):
        self.cursor.execute("""SELECT * FROM messages""")
        return [
            (
                id,
                session_id,
                user_query,
                k,
                fetch_k,
                lambda_mult,
                score_threshold,
                response,
                source_docs,
                timestamp,
            )
            for id, session_id, user_query, k, fetch_k, lambda_mult, score_threshold, response, source_docs, timestamp in self.cursor.fetchall()
        ]

    def fetch_sessions(self):
        self.cursor.execute(
            """SELECT user_query, session_id FROM messages GROUP BY session_id ORDER BY timestamp DESC"""
        )

        return [[row[0], row[1]] for row in self.cursor.fetchall()]

    def fetch_messages_by_session(self, session_id):
        # 주어진 세션 ID에 해당하는 메시지를 조회하는 쿼리를 실행합니다.
        self.cursor.execute(
            """SELECT id, user_query,kwargs_json, response, source_docs, timestamp 
                               FROM messages WHERE session_id = ? ORDER BY timestamp""",
            (session_id,),
        )
        # 결과를 가져와서 파이썬 객체로 변환합니다.
        messages = self.cursor.fetchall()
        # JSON 형식의 source_docs를 파이썬 객체로 변환합니다.
        return [
            (
                id,
                user_query,
                k,
                fetch_k,
                lambda_mult,
                score_threshold,
                kwargs_json,
                response,
                json.loads(source_docs),
                timestamp,
            )
            for id, user_query, k, fetch_k, lambda_mult, score_threshold, kwargs_json, response, source_docs, timestamp in messages
        ]

    def close(self):
        self.conn.close()
