import json
import sqlite3

class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS messages
                             (id INTEGER PRIMARY KEY AUTOINCREMENT,
                              user_query TEXT,
                              response TEXT,
                              source_docs TEXT,
                              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        self.conn.commit()


    def insert_message(self, user_query, response, k):
        # Serialize the source_docs list to a JSON string
        source_docs_json = json.dumps([
            {'page_content': doc.page_content, 'metadata': doc.metadata}
            for doc in k
        ],ensure_ascii=False)

        # Insert the data into the database, using the JSON string for source_docs
        self.cursor.execute('''INSERT INTO messages (user_query, response, source_docs)
                               VALUES (?, ?, ?)''', (user_query, response, source_docs_json))
        self.conn.commit()

    def fetch_all_messages(self):
        self.cursor.execute('''SELECT * FROM messages''')
        return [(id, user_query, response, json.loads(source_docs), timestamp)
                for id, user_query, response, source_docs, timestamp in self.cursor.fetchall()]

    def close(self):
        self.conn.close()
