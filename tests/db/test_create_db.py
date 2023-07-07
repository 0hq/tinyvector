import sqlite3
from database import DB
from models.db import TableMetadata
import pytest
import os


def test_create_table(tmp_path):
    db_path = tmp_path / "test.db"
    table_name = "test_table"

    db_config = TableMetadata(
        allow_index_updates=False,
        dimension=2,
        index_type="brute_force",
        is_index_active=True,
        normalize=True,
        use_uuid=False,
    )
    try:
        test_db = DB(db_path)
        test_db.create_table_and_index(table_name, db_config)

        # Check if the table has been created
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None, f"Table {table_name} not created"

        # Validate that trying to create the same table again raises an error
        with pytest.raises(ValueError, match=f"Table {table_name} already exists"):
            test_db.create_table_and_index(table_name, db_config)
    finally:
        os.remove(db_path)
