import pickle

from flask import Flask, jsonify, request

from db import SQLiteDatabase

app = Flask(__name__)

# Initialize database
db = SQLiteDatabase('my_database.db')

@app.route('/items', methods=['GET'])
def get_items():
    items = db.get_all()
    return jsonify(items)

@app.route('/items/<int:id>', methods=['GET'])
def get_item(id):
    item = db.get_by_id(id)
    if item is None:
        return jsonify({'error': 'Item not found'}), 404
    return jsonify(item)

@app.route('/items', methods=['POST'])
def create_item():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    try:
        text = request.json['text']
        embedding = request.json['embedding']
        extra_data = request.json.get('extra_data')
        id = db.insert(text, embedding, extra_data)
        return jsonify({"id": id}), 201
    except KeyError as e:
        return jsonify({"error": f"Missing field in JSON: {e}"}), 400

@app.route('/items/<int:id>', methods=['PUT'])
def update_item(id):
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    try:
        text = request.json['text']
        embedding = request.json['embedding']
        extra_data = request.json.get('extra_data')
        id = db.update(id, text, embedding, extra_data)
        return jsonify({"id": id}), 200
    except KeyError as e:
        return jsonify({"error": f"Missing field in JSON: {e}"}), 400

@app.route('/items/<int:id>', methods=['DELETE'])
def delete_item(id):
    db.delete(id)
    return jsonify({}), 204

if __name__ == '__main__':
    app.run(debug=True)
