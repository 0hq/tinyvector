import logging
import os
from functools import wraps

import jwt
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from vectordb import DB

logging.basicConfig(filename='logs/app.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

app = Flask(__name__)
load_dotenv()
db = DB('database/database.db')

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
app.logger.addHandler(stream_handler)

def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = None

        if app.debug:
            return f(*args, **kwargs)

        if 'Authorization' in request.headers:
            token = request.headers['Authorization']

        if not token:
            app.logger.warning('Token is missing!')
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            _ = jwt.decode(token, os.getenv('JWT_SECRET'), algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            app.logger.error('Token is expired!')
            return jsonify({'message': 'Token is expired!'}), 401
        except jwt.InvalidTokenError:
            app.logger.error('Token is invalid!')
            return jsonify({'message': 'Token is invalid!'}), 401

        return f(*args, **kwargs)

    return decorator

@app.route('/status', methods=['GET'])
def status():
    app.logger.info('Status check performed')
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
        app.logger.info(f'Table {table_name} created successfully')
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        app.logger.error(f'Error while creating table {table_name}: {str(e)}')
        return jsonify({'error': str(e)}), 400

@app.route('/delete_table', methods=['DELETE'])
@token_required
def delete_table():
    data = request.get_json()
    table_name = data.get('table_name')
    try:
        db.delete_table(table_name)
        app.logger.info(f'Table {table_name} deleted successfully')
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        app.logger.error(f'Error while deleting table {table_name}: {str(e)}')
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
        app.logger.info(f'Item {id} inserted successfully into table {table_name}')
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        app.logger.error(f'Error while inserting item {id}: {str(e)}')
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
        app.logger.info(f'Query performed successfully on table {table_name}')
        return jsonify({'items': items}), 200
    except Exception as e:
        app.logger.error(f'Error while performing query on table {table_name}: {str(e)}')
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
        app.logger.info(f'Index created successfully on table {table_name}')
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        app.logger.error(f'Error while creating index on table {table_name}: {str(e)}')
        return jsonify({'error': str(e)}), 400

@app.route('/delete_index', methods=['DELETE'])
@token_required
def delete_index():
    data = request.get_json()
    table_name = data.get('table_name')
    try:
        db.delete_index(table_name)
        app.logger.info(f'Index deleted successfully on table {table_name}')
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        app.logger.error(f'Error while deleting index on table {table_name}: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.logger.info('Starting server...')
    app.run(host='0.0.0.0', port=5000, debug=True)
