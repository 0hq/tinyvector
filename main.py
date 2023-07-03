from flask import Flask, request, jsonify
from code import DB 
import numpy as np
import jwt
from functools import wraps
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()
db = DB('database.db')  # initialize DB with database.db sqlite file

def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):

        token = None

        if 'Authorization' in request.headers:
            token = request.headers['Authorization']

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            payload = jwt.decode(token, os.getenv('SECRET_KEY'), algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token is expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid!'}), 401

        return f(*args, **kwargs)

    return decorator

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'success'}), 200

@app.route('/create_table', methods=['POST'])
@token_required
def create_table():
    data = request.get_json()
    table_name = data.get('table_name')
    dimension = data.get('dimension')
    use_uuid = data.get('use_uuid', False)
    try:
        db.create_table(table_name, dimension, use_uuid)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/delete_table', methods=['DELETE'])
@token_required
def delete_table():
    data = request.get_json()
    table_name = data.get('table_name')
    try:
        db.delete_table(table_name)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/insert', methods=['POST'])
@token_required
def insert():
    data = request.get_json()
    table_name = data.get('table_name')
    id = data.get('id') # None for UUID tables, otherwise a string
    embedding = data.get('embedding')
    content = data.get('content', None)
    defer_index_update = data.get('defer_index_update', False)
    try:
        embedding = np.array(embedding)
        db.insert(table_name, id, embedding, content, defer_index_update)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/query', methods=['POST'])
@token_required # Remove this if you want to allow unauthenticated queries to your database
def query():
    data = request.get_json()
    table_name = data.get('table_name')
    query = data.get('query')
    k = data.get('k')
    try:
        query = np.array(query)
        items = db.query(table_name, query, k)
        return jsonify({'items': items}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/create_index', methods=['POST'])
@token_required
def create_index():
    data = request.get_json()
    table_name = data.get('table_name')
    index_type = data.get('index_type', 'brute_force')
    normalize = data.get('normalize', True) # True for cosine similarity, False for dot product
    allow_index_updates = data.get('allow_index_updates', None)
    n_components = data.get('n_components', None)
    try:
        db.create_index(table_name, index_type, normalize, allow_index_updates, n_components)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/delete_index', methods=['DELETE'])
@token_required
def delete_index():
    data = request.get_json()
    table_name = data.get('table_name')
    try:
        db.delete_index(table_name)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
