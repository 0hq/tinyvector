from flask import Flask, request, jsonify
from code import DB 
import numpy as np

app = Flask(__name__)

db = DB('database.db')  # initialize DB with database.db sqlite file

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'success'}), 200

@app.route('/create_table', methods=['POST'])
def create_table():
    data = request.get_json()
    table_name = data.get('table_name')
    dimension = data.get('dimension')
    try:
        db.create_table(table_name, dimension)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/delete_table', methods=['DELETE'])
def delete_table():
    data = request.get_json()
    table_name = data.get('table_name')
    try:
        db.delete_table(table_name)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/insert', methods=['POST'])
def insert():
    data = request.get_json()
    table_name = data.get('table_name')
    id = data.get('id')
    embedding = data.get('embedding')
    defer_index_update = data.get('defer_index_update', False)
    try:
        embedding = np.array(embedding)
        db.insert(table_name, id, embedding, defer_index_update)
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/query', methods=['POST'])
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
