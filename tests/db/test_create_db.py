import sqlite3
from database import DB
from models.db import DatabaseInfo, TableCreationBody, TableMetadata
import pytest
import os


@pytest.fixture
def create_db(tmp_path):
    db_path = tmp_path / "test.db"
    table_name = "test_table"

    db_config = TableCreationBody(
        table_name=table_name,
        allow_index_updates=False,
        dimension=42,
        index_type="brute_force",
        is_index_active=True,
        normalize=True,
        use_uuid=False,
    )
    test_db = DB(db_path)
    test_db.create_table_and_index(db_config)
    yield test_db, table_name, db_config, db_path
    os.remove(db_path)


def test_create_table_again_raises_error(create_db):
    test_db, table_name, db_config, db_path = create_db
    with pytest.raises(ValueError, match=f"Table {table_name} already exists"):
        test_db.create_table_and_index(db_config)


def test_loads_correct_metadata_on_startup(create_db):
    test_db, table_name, db_config, db_path = create_db
    test_db_2 = DB(db_path)
    info = test_db_2.info()
    print(info)
    assert table_name in info.tables, f"Table {table_name} not found"
    assert TableMetadata(**db_config.model_dump()) == info.tables[table_name]
    assert info.num_tables == 1, "Number of tables should be 1"


def test_delete_table(create_db):
    test_db, table_name, db_config, db_path = create_db
    test_db.delete_table(table_name)
    info = test_db.info()
    assert table_name not in info.tables, f"Table {table_name} should not be found"
    assert info.num_tables == 0, "Number of tables should be 0"


def test_create_index(create_db):
    test_db, table_name, db_config, db_path = create_db
    # TODO
    pass
