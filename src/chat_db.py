import sqlite3
from datetime import datetime
import os
import json

# Use an absolute path or a path relative to the project root
# ensuring the database file is created in a reliable location
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chat_history.db')

# Initialize database and tables if not exist
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        name TEXT,
        created_at TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT,
        sender TEXT,
        content TEXT,
        timestamp TEXT,
        metadata TEXT,
        FOREIGN KEY(chat_id) REFERENCES chats(id)
    )''')
    conn.commit()

    # Ensure metadata column exists for older DBs
    c.execute("PRAGMA table_info(messages)")
    cols = [r[1] for r in c.fetchall()]
    if 'metadata' not in cols:
        try:
            c.execute('ALTER TABLE messages ADD COLUMN metadata TEXT')
            conn.commit()
        except Exception:
            # ignore if cannot alter (older SQLite versions)
            pass
    conn.close()

# Add a new chat
def add_chat(chat_id, name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Check if chat already exists to prevent unique constraint errors
    c.execute('SELECT id FROM chats WHERE id = ?', (chat_id,))
    if c.fetchone() is None:
        c.execute('INSERT INTO chats (id, name, created_at) VALUES (?, ?, ?)',
                  (chat_id, name, datetime.now().isoformat()))
        conn.commit()
    conn.close()

# Add a message to a chat
def add_message(chat_id, sender, content, metadata: dict = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    meta_json = None
    if metadata is not None:
        try:
            meta_json = json.dumps(metadata)
        except Exception:
            meta_json = json.dumps({})

    c.execute('INSERT INTO messages (chat_id, sender, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)',
              (chat_id, sender, content, datetime.now().isoformat(), meta_json))
    conn.commit()
    conn.close()

# Get all chats
def get_chats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, created_at FROM chats ORDER BY created_at DESC')
    chats = c.fetchall()
    conn.close()
    return chats

# Get messages for a chat
def get_messages(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT sender, content, timestamp, metadata FROM messages WHERE chat_id=? ORDER BY timestamp', (chat_id,))
    rows = c.fetchall()
    conn.close()

    messages = []
    for sender, content, timestamp, metadata in rows:
        meta = None
        if metadata:
            try:
                meta = json.loads(metadata)
            except Exception:
                meta = None
        messages.append((sender, content, timestamp, meta))

    return messages

def get_chat_by_id(chat_id):
    """Return chat tuple (id, name, created_at) for the given chat_id or None."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, created_at FROM chats WHERE id = ?', (chat_id,))
    chat = c.fetchone()
    conn.close()
    return chat

# Rename a chat
def rename_chat(chat_id, new_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE chats SET name=? WHERE id=?', (new_name, chat_id))
    conn.commit()
    conn.close()

# Delete a chat and its messages
def delete_chat(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM messages WHERE chat_id=?', (chat_id,))
    c.execute('DELETE FROM chats WHERE id=?', (chat_id,))
    conn.commit()
    conn.close()

# Call init_db() at module load
init_db()