import logging
import os
from functools import wraps

import jwt
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, g
from pydantic import BaseModel

from database import DB
from flask_pydantic_spec import FlaskPydanticSpec
from flask_pydantic_spec import Response

from models.db import (
    DatabaseInfo,
    IndexCreationBody,
    IndexDeletionBody,
    ItemInsertionBody,
    TableCreationBody,
    TableDeletionBody,
    TableMetadata,
    TableQueryObject,
    TableQueryResult,
)
from models.response import ErrorMessage
from utils.pydantic import pydantic_to_dict


logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s",
)
load_dotenv()

# If JWT_SECRET is not set, the application will run in debug mode
if os.getenv("JWT_SECRET") is None:
    os.environ["FLASK_ENV"] = "development"


app = Flask(__name__)
api = FlaskPydanticSpec(
    "Tiny Vector Database",
)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
app.logger.addHandler(stream_handler)

DATABASE_PATH = os.environ.get("DATABASE_PATH", "cache/database.db")


def get_db():
    if "db" not in g:
        g.db = DB(DATABASE_PATH)
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)


def token_required(f):
    """
    Decorator function to enforce token authentication for specific routes.
    """

    @wraps(f)
    def decorator(*args, **kwargs):
        token = None

        if app.debug:
            return f(*args, **kwargs)

        if "Authorization" in request.headers:
            token = request.headers["Authorization"]

        if not token:
            app.logger.warning("Token is missing!")
            return jsonify({"message": "Token is missing!"}), 401

        JWT_SECRET = os.getenv("JWT_SECRET")
        if JWT_SECRET is None:
            raise Exception("JWT_SECRET is not set!")

        try:
            _ = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            app.logger.error("Token is expired!")
            return jsonify({"message": "Token is expired!"}), 401
        except jwt.InvalidTokenError:
            app.logger.error("Token is invalid!")
            return jsonify({"message": "Token is invalid!"}), 401

        return f(*args, **kwargs)

    return decorator


class SuccessMessage(BaseModel):
    status: str = "success"


@app.route("/status", methods=["GET"])
@api.validate(body=None, resp=Response(HTTP_200=SuccessMessage), tags=["API"])
@pydantic_to_dict
def status():
    """
    Route to check the status of the application.
    """
    app.logger.info("Status check performed")
    return SuccessMessage(status="success"), 200


@app.route("/info", methods=["GET"])
@token_required
@api.validate(
    body=None, resp=Response(HTTP_200=DatabaseInfo, HTTP_400=ErrorMessage), tags=["API"]
)
@pydantic_to_dict
def info():
    """
    Route to get information about the database.
    """
    try:
        info = get_db().info()
        app.logger.info("Info retrieved successfully")
        return info, 200
    except Exception as e:
        app.logger.error(f"Error while retrieving info: {str(e)}")
        return ErrorMessage(error=f"Error while retrieving info: {str(e)}"), 400


@app.route("/create_table", methods=["POST"])
@token_required
@api.validate(
    body=TableCreationBody,
    resp=Response(HTTP_200=SuccessMessage, HTTP_400=ErrorMessage),
    tags=["DB"],
)
@pydantic_to_dict
def create_table():
    """
    Route to create a table in the database.
    If use_uuid is True, the table will use UUIDs as IDs, and the IDs provided in the insert route are not allowed.
    If use_uuid is False, the table will require strings as IDs.
    """

    data = request.get_json()
    body = TableCreationBody(**data)
    try:
        get_db().create_table_and_index(body)
        app.logger.info(f"Table {body.table_name} created successfully")
        return (
            SuccessMessage(status=f"Table {body.table_name} created successfully"),
            200,
        )
    except Exception as e:
        app.logger.error(f"Error while creating table {body.table_name}: {str(e)}")
        return ErrorMessage(error=str(e)), 400


@app.route("/delete_table", methods=["DELETE"])
@api.validate(
    body=TableDeletionBody,
    resp=Response(HTTP_200=SuccessMessage, HTTP_400=ErrorMessage),
    tags=["DB"],
)
@token_required
@pydantic_to_dict
def delete_table():
    """
    Route to permanently delete a table and its data from the database.
    This will also delete the index associated with the table.
    """
    data = request.get_json()
    body = TableDeletionBody(**data)
    table_name = body.table_name
    try:
        get_db().delete_table(table_name)
        app.logger.info(f"Table {table_name} deleted successfully")
        return SuccessMessage(status=f"Table {table_name} deleted successfully"), 200
    except Exception as e:
        app.logger.error(f"Error while deleting table {table_name}: {str(e)}")
        return (
            ErrorMessage(status=f"Error while deleting table {table_name}: {str(e)}"),
            400,
        )


@app.route("/insert", methods=["POST"])
@api.validate(
    body=ItemInsertionBody,
    resp=Response(HTTP_200=SuccessMessage, HTTP_400=ErrorMessage),
    tags=["DB"],
)
@token_required
@pydantic_to_dict
def insert():
    """
    Route to insert an item into a table in the database.
    Requires a previously generated vector embedding of the right dimension (set when creating the table).
    If use_uuid was set to True when creating the table, the ID will be generated automatically. Providing an ID in the request will result in an error.
    If use_uuid was set to False, the ID must be provided as a string.
    Defer index update can be set to True to stop the index from being updated after the insert. This only works for brute force index, as other indexes can't be efficiently updated after creation.
    """
    data = request.get_json()
    body = ItemInsertionBody(**data)

    table_name = body.table_name
    id = body.id
    embedding = body.embedding
    content = body.content
    defer_index_update = body.defer_index_update
    try:
        embedding = np.array(embedding)
        get_db().insert(table_name, id, embedding, content, defer_index_update)
        app.logger.info(f"Item {id} inserted successfully into table {table_name}")
        return (
            SuccessMessage(
                status=f"Item {id} inserted successfully into table {table_name}"
            ),
            200,
        )
    except Exception as e:
        app.logger.error(f"Error while inserting item {id}: {str(e)}")
        return (
            ErrorMessage(status=f"Error while inserting item {id}: {str(e)}"),
            400,
        )


@app.route("/query", methods=["POST"])
@token_required
@api.validate(
    body=TableQueryObject,
    resp=Response(HTTP_200=TableQueryResult, HTTP_400=ErrorMessage),
    tags=["DB"],
)
@pydantic_to_dict
def query():
    """
    Route to perform a query on a table in the database.
    Requires a previously generated query vector embedding of the right dimension (set when creating the table).
    K is the number of items to return.
    """
    body = TableQueryObject(**request.get_json())

    table_name = body.table_name
    query = body.query
    k = body.k
    try:
        query = np.array(query)
        items = get_db().query(table_name, query, k)
        app.logger.info(f"Query performed successfully on table {table_name}")
        print(items)
        return TableQueryResult(items=items), 200
    except Exception as e:
        app.logger.error(
            f"Error while performing query on table {table_name}: {str(e)}"
        )
        return (
            ErrorMessage(
                error=f"Error while performing query on table {table_name}: {str(e)}"
            ),
            400,
        )


@app.route("/create_index", methods=["POST"])
@api.validate(
    body=IndexCreationBody,
    tags=["DB"],
    resp=Response(HTTP_200=SuccessMessage, HTTP_400=ErrorMessage),
)
@token_required
@pydantic_to_dict
def create_index():
    """
    Route to create an index on a table in the database.
    Index type can be 'brute_force' or 'pca'.
    PCA index is a dimensionality reduction index, and n_components is the number of components to keep and must be less than the dimension of the table. See README for more details.
    If normalize is True, cosine similarity will be used. If False, dot product will be used.
    If allow_index_updates is True, the index will be updated after each insert. This only works for brute force index, as other indexes can't be efficiently updated after creation.
    If you want to update index contents from a non-updatable index (PCA, others), the reccomended method is to delete and create a new one.
    """
    body = IndexCreationBody(**request.get_json())

    table_name = body.table_name
    index_type = body.index_type
    normalize = body.normalize
    allow_index_updates = body.allow_index_updates
    n_components = body.n_components

    try:
        get_db().create_index(
            table_name, index_type, normalize, allow_index_updates, n_components
        )
        app.logger.info(f"Index created successfully on table {table_name}")
        return (
            SuccessMessage(status=f"Index created successfully on table {table_name}"),
            200,
        )
    except Exception as e:
        app.logger.error(f"Error while creating index on table {table_name}: {str(e)}")
        return (
            ErrorMessage(
                error=f"Error while creating index on table {table_name}: {str(e)}"
            ),
            400,
        )


@app.route("/delete_index", methods=["DELETE"])
@token_required
@api.validate(
    body=IndexDeletionBody,
    tags=["DB"],
    resp=Response(HTTP_200=SuccessMessage, HTTP_400=ErrorMessage),
)
@pydantic_to_dict
def delete_index():
    """
    Route to delete an index from a table in the database.
    This will not delete the table or its data.
    """
    body = IndexDeletionBody(**request.get_json())
    table_name = body.table_name
    try:
        get_db().delete_index(table_name)
        app.logger.info(f"Index deleted successfully on table {table_name}")
        return (
            SuccessMessage(status=f"Index deleted successfully on table {table_name}"),
            200,
        )
    except Exception as e:
        app.logger.error(f"Error while deleting index on table {table_name}: {str(e)}")
        return (
            ErrorMessage(
                error=f"Error while deleting index on table {table_name}: {str(e)}"
            ),
            400,
        )


if __name__ == "__main__":
    app.logger.info("Starting server...")

    api.register(app)
    PORT = 5234
    app.logger.info(
        f"\nSuccesfully Generated Documentation :) \n\n- Redoc: http://localhost:{PORT}/apidoc/redoc \n- Swagger: http://localhost:{PORT}/apidoc/swagger"
    )
    app.run(host="0.0.0.0", port=PORT, debug=True)
