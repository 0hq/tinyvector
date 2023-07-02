from flask import Flask, request, jsonify
from flask_restful import Resource, Api, abort
import os
import pickle
import numpy as np
from db import SQLiteDatabase

app = Flask(__name__)
api = Api(app)

DATABASES_PATH = 'databases.pickle'
databases = {}

def load_databases():
    if os.path.exists(DATABASES_PATH):
        with open(DATABASES_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def save_databases():
    with open(DATABASES_PATH, 'wb') as f:
        pickle.dump(databases, f)

databases = load_databases()

class DatabaseResource(Resource):
    def get(self, db_name, include_vectors=False):
        # Retrieve a database
        if db_name not in databases:
            abort(404, message=f"Database {db_name} doesn't exist")
        db = databases[db_name]
        return jsonify({"db_name": db_name, "dimensions": db.dimensions, "indexes": list(db.indexes.keys()), "num_vectors": len(db), "vectors": db.get_all() if include_vectors else None})

    def post(self, db_name):
        # Create a new database
        dimensions = request.json.get('dimensions', 300)  # Use a default value if not provided
        databases[db_name] = SQLiteDatabase(db_name, dimensions)
        return {"message": f"Database {db_name} created"}, 201

    def delete(self, db_name):
        if db_name in databases:
            del databases[db_name]
            save_databases()
            return {"message": f"Database {db_name} deleted"}
        else:
            abort(404, message=f"Database {db_name} doesn't exist")



class IndexResource(Resource):
    def get(self, db_name, index_name, include_vectors=False):
        # Retrieve an index
        db = databases.get(db_name)
        if not db:
            abort(404, message=f"Database {db_name} doesn't exist")
        index = db.indexes.get(index_name)
        if not index:
            abort(404, message=f"Index {index_name} doesn't exist")
        return {"index_name": index_name, "type": index.type, "num_vectors": index.num_vectors, "dimensions": index.dimension, "vectors": index.get_all() if include_vectors else None}

    def post(self, db_name, index_name):
        # Create a new index
        db = databases.get(db_name)
        if not db:
            abort(404, message=f"Database {db_name} doesn't exist")
        index_type = request.json.get('index_type', 'Base')
        db.create_index(index_type, index_name)
        return {"message": f"Index {index_name} created"}, 201

    def delete(self, db_name, index_name):
        # Delete an index
        db = databases.get(db_name)
        if not db:
            abort(404, message=f"Database {db_name} doesn't exist")
        db.delete_index(index_name)
        return {"message": f"Index {index_name} deleted"}


class QueryResource(Resource):
    def post(self, db_name, index_name):
        # Query an index
        db = databases.get(db_name)
        if not db:
            abort(404, message=f"Database {db_name} doesn't exist")
        index = db.indexes.get(index_name)
        if not index:
            abort(404, message=f"Index {index_name} doesn't exist")
        query_embedding = np.array(request.json.get('query_embedding'))
        k = request.json.get('k', 10)
        results = index.get_similarity(query_embedding, k)
        return jsonify(results)
    
class ItemResource(Resource):
    def get(self, db_name, index_name, item_id):
        db = databases.get(db_name)
        if not db:
            abort(404, message=f"Database {db_name} doesn't exist")
        index = db.indexes.get(index_name)
        if not index:
            abort(404, message=f"Index {index_name} doesn't exist")
        item = index.get_item(item_id) 
        if not item:
            abort(404, message=f"Item {item_id} doesn't exist")
        return jsonify(item)

    def post(self, db_name, index_name):
        db = databases.get(db_name)
        if not db:
            abort(404, message=f"Database {db_name} doesn't exist")
        index = db.indexes.get(index_name)
        if not index:
            abort(404, message=f"Index {index_name} doesn't exist")
        if not index.allow_updates:
            abort(405, message=f"Index {index_name} doesn't allow updates")
        item_id = request.json.get('item_id')
        vector = np.array(request.json.get('vector'))
        index.add_item(item_id, vector)
        return {"message": f"Item {item_id} added"}, 201

# Add URL routes
api.add_resource(DatabaseResource, '/database/<string:db_name>')
api.add_resource(IndexResource, '/database/<string:db_name>/index/<string:index_name>')
api.add_resource(QueryResource, '/database/<string:db_name>/index/<string:index_name>/query')
api.add_resource(ItemResource, '/database/<string:db_name>/index/<string:index_name>/item', '/database/<string:db_name>/index/<string:index_name>/item/<string:item_id>')

if __name__ == '__main__':
    app.run(debug=True)
