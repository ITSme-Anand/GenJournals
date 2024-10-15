// database.js
import * as SQLite from 'expo-sqlite';

const db = SQLite.openDatabaseAsync('chatbot.db');



export async function createTables() {
  
  await db.execAsync(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY NOT NULL,
      sender TEXT NOT NULL,
      message TEXT NOT NULL,
      timestamp TEXT NOT NULL
    );
  `);
}


export async function insertMessage(sender, message) {
  const timestamp = new Date().toISOString();
  const result = await db.execAsync(
    'INSERT INTO messages (sender, message, timestamp) VALUES (?, ?, ?)',
    sender,
    message,
    timestamp
  );
  return result.lastInsertRowId;
}

export async function fetchMessages() {
  const messages = await db.getAllAsync('SELECT * FROM messages');
  return messages;
}

export async function deleteMessage(id) {
  await db.execAsync('DELETE FROM messages WHERE id = ?', id);
}

