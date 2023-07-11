import json
import os
import random
import time
from unittest.mock import patch

import jwt as pyjwt
import pytest

from server.app import SuccessMessage, app
from server.types.model_response import ErrorMessage
from tinyvector.database import DB
from tinyvector.types.model_db import (DatabaseInfo, ItemInsertionBody,
                                       TableCreationBody, TableDeletionBody,
                                       TableMetadata, TableQueryObject)

test_table = "test_table"
jwt_secret = "testing_config"
db_path = "test_endpoint_db.db"
new_table = TableMetadata(
    table_name=test_table,
    allow_index_updates=True,
    dimension=42,
    index_type="brute_force",
    is_index_active=True,
    normalize=True,
    use_uuid=False,
)


def generate_client(app):
    os.environ["JWT_SECRET"] = jwt_secret
    encoded_jwt = pyjwt.encode(
        {"payload": "random payload"}, jwt_secret, algorithm="HS256"
    )
    headers = {"Authorization": encoded_jwt}

    client = app.test_client()
    client.environ_base.update(headers)
    return client, headers


@pytest.fixture(scope="session")
def create_test_db():
    test_db = DB(path=db_path)
    test_db.create_table_and_index(new_table)
    yield test_db
    os.remove(db_path)


def test_token_authentication(create_test_db, mocker):
    """
    This validates that the info endpoint correctly captures the state of the DB at a specific snapshot in time.
    """
    # Mock the DB object
    mocker.patch("server.app.get_db", return_value=create_test_db)

    test_client, headers = generate_client(app)

    # /info
    with test_client as client:
        response = client.get("/info")
        data = json.loads(response.data)

        assert response.status_code == 401
        assert data == {"message": "Token is missing!"}

        response = client.get("/info", headers={"Authorization": "random_token"})
        data = json.loads(response.data)
        assert response.status_code == 401
        assert data == {"message": "Token is invalid!"}

        expired_jwt = pyjwt.encode(
            {"payload": "random payload", "exp": int(time.time()) - 10},
            jwt_secret,
            algorithm="HS256",
        )
        response = client.get("/info", headers={"Authorization": expired_jwt})
        data = json.loads(response.data)
        assert response.status_code == 401
        assert data == {"message": "Token is expired!"}


def test_info(create_test_db, mocker):
    """
    This validates that the info endpoint correctly captures the state of the DB at a specific snapshot in time.
    """
    # Mock the DB object
    mocker.patch("server.app.get_db", return_value=create_test_db)

    test_client, headers = generate_client(app)

    # /info
    with test_client as client:
        response = client.get("/info", headers=headers)
        data = json.loads(response.data)

        assert response.status_code == 200

        expected_response = DatabaseInfo(
            num_tables=1,
            num_indexes=1,
            tables={},
            indexes=["test_table brute_force_mutable"],
        )
        expected_response.tables[test_table] = new_table

        # Replace the expected data with the actual data you expect from DB.info()
        assert data == expected_response.dict()


def test_status_endpoint(create_test_db, mocker):
    """
    This validates that the info endpoint correctly captures the state of the DB at a specific snapshot in time.
    """
    # Mock the DB object
    mocker.patch("server.app.get_db", return_value=create_test_db)

    test_client, headers = generate_client(app)

    # /info
    with test_client as client:
        response = client.get("/status")
        data = json.loads(response.data)

        assert response.status_code == 200

        # Replace the expected data with the actual data you expect from DB.info()
        assert data == SuccessMessage(status="success").dict()


def test_create_endpoint(create_test_db, mocker):
    """
    This validates that we can succesfully create a table object and have the database be updated and populated with the new table.
    """
    # Mock the DB object
    mocker.patch("server.app.get_db", return_value=create_test_db)

    test_client, headers = generate_client(app)

    # /info
    with test_client as client:
        table_2 = TableCreationBody(
            table_name="table_testing_2",
            allow_index_updates=True,
            dimension=42,
            index_type="brute_force",
            is_index_active=True,
            normalize=True,
            use_uuid=False,
        )

        response = client.post(
            "/create_table", json=table_2.dict(), headers=headers
        )

        data = json.loads(response.data)

        assert response.status_code == 200
        assert (
            data
            == SuccessMessage(
                status=f"Table {table_2.table_name} created successfully"
            ).dict()
        )

        response = client.get("/info", headers=headers)
        data = json.loads(response.data)

        assert response.status_code == 200
        assert data["num_tables"] == 2
        assert data["num_indexes"] == 2
        assert data["tables"][table_2.table_name] == table_2.dict()

        # We then create a new db instance which loads metadata from scratch based on sqlite3 database and validate that it has the same data

        new_db = DB(path=db_path, debug=True)
        new_instance_data = new_db.info()
        assert new_instance_data.dict() == data


def test_delete_endpoint(create_test_db, mocker):
    """
    This validates the delete endpoint
    """
    # Mock the DB object
    mocker.patch("server.app.get_db", return_value=create_test_db)

    test_client, headers = generate_client(app)

    with test_client as client:
        # We previously created a new table in a previous test, let's make sure that the changes have been applied.

        response = client.get("/info", headers=headers)
        body = json.loads(response.data)
        assert response.status_code == 200
        assert body["num_tables"] == 2
        assert body["num_indexes"] == 2

        # Try to delete an invalid table
        body = TableDeletionBody(
            table_name="fake_table",
        )

        response = client.delete(
            "/delete_table", json=body.dict(), headers=headers
        )
        data = json.loads(response.data)
        assert response.status_code == 400
        assert (
            data
            == ErrorMessage(
                status=f"Error while deleting table fake_table: Table fake_table does not exist"
            ).dict()
        )

        # Try to delete a valid table
        body = TableDeletionBody(
            table_name="table_testing_2",
        )
        response = client.delete(
            "/delete_table", json=body.dict(), headers=headers
        )
        data = json.loads(response.data)
        assert response.status_code == 200
        assert (
            data
            == SuccessMessage(
                status=f"Table {body.table_name} deleted successfully"
            ).dict()
        )

        # Validate that the table has been deleted
        response = client.get("/info", headers=headers)
        data = json.loads(response.data)

        assert response.status_code == 200
        assert data["num_tables"] == 1
        assert data["num_indexes"] == 1

        # We then create a new db instance which loads metadata from scratch based on sqlite3 database and validate that it has the same data

        new_db = DB(path=db_path, debug=True)
        new_instance_data = new_db.info()
        assert new_instance_data.dict() == data


def test_insert_endpoint(create_test_db, mocker):
    """
    This validates the delete endpoint
    """
    # Mock the DB object
    mocker.patch("server.app.get_db", return_value=create_test_db)

    test_client, headers = generate_client(app)

    with test_client as client:
        values = 200
        for i in range(values):
            item = ItemInsertionBody(
                table_name=test_table,
                id=f"Item {i}",
                embedding=[random.randint(0, values) for i in range(42)],
                content=f"Item {i} content",
                defer_index_update=False,
            )
            response = client.post("/insert", json=item.dict(), headers=headers)
            data = json.loads(response.data)
            assert response.status_code == 200
            assert (
                data["status"]
                == f"Item Item {i} inserted successfully into table {test_table}"
            )

        # We now validate that our table has a total of 10 items by calling the /query endpoint
        body = TableQueryObject(
            table_name=test_table, query=[0 for i in range(42)], k=values
        )
        response = client.post("/query", json=body.dict(), headers=headers)
        data = json.loads(response.data)

        assert response.status_code == 200
        assert len(data["items"]) == values
