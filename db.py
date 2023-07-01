import pickle
import sqlite3


class SQLiteDatabase:
    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)
        self.conn.text_factory = bytes  # This makes sqlite3 return blobs as bytes objects.

        # Define fields
        fields_sql = "id INTEGER PRIMARY KEY, text TEXT NOT NULL, embedding BLOB NOT NULL"

        self.conn.execute(f"CREATE TABLE IF NOT EXISTS my_table ({fields_sql})")

    def insert(self, text, embedding):
        # Convert data to bytes using pickle
        embedding_as_bytes = pickle.dumps(embedding)

        try:
            cursor = self.conn.execute(
                "INSERT INTO my_table (text, embedding) VALUES (?, ?)",
                (text, embedding_as_bytes)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Insertion error: {e}")
            return None

        return cursor.lastrowid  # Return the ID of the new record

    def get_by_id(self, id):
        try:
            cursor = self.conn.execute(
                "SELECT * FROM my_table WHERE id = ?",
                (id,)
            )
        except sqlite3.Error as e:
            print(f"Selection error: {e}")
            return None

        row = cursor.fetchone()
        if row is None:
            return None

        # Parse the result
        return self._parse_row(row)

    def get_all(self):
        try:
            cursor = self.conn.execute("SELECT * FROM my_table")
        except sqlite3.Error as e:
            print(f"Selection error: {e}")
            return None

        result = []
        for row in cursor:
            result.append(self._parse_row(row))

        return result

    def update(self, id, text, embedding):
        # Convert data to bytes using pickle
        embedding_as_bytes = pickle.dumps(embedding)

        try:
            self.conn.execute(
                "UPDATE my_table SET text = ?, embedding = ? WHERE id = ?",
                (text, embedding_as_bytes, id)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Update error: {e}")
            return None

        return id  # Return the ID of the updated record

    def delete(self, id):
        try:
            self.conn.execute(
                "DELETE FROM my_table WHERE id = ?",
                (id,)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Deletion error: {e}")
            return False

        return True

    def _parse_row(self, row):
        # This helper function parses a row and returns it as a dictionary.
        id, text, embedding_as_bytes = row
        embedding = pickle.loads(embedding_as_bytes)

        return {'id': id, 'text': text, 'embedding': embedding}
