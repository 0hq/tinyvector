import sqlite3
import pickle
import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
import json

class AbstractIndex(ABC):
    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def get_by_id(self, id):
        pass

    @abstractmethod
    def get_similarity(self, query, k, normalize=True, dataset_is_normalized=False, useGPU=False):
        pass


class BaseIndex(AbstractIndex):
    def __init__(self, embeddings, ids):
        self.embeddings = embeddings
        self.ids = ids

    def get_all(self):
        return {'id': self.ids, 'embedding': self.embeddings}

    def get_by_id(self, id):
        index = self.ids.index(id)
        return {'id': id, 'embedding': self.embeddings[index]}

    def normalize(self, vectors):
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Time complexity: O(n_vectors + k log k)
    def get_similarity(self, query, k, normalize=True, dataset_is_normalized=False, useGPU=False):
        """
        Find the top-k most similar items to a query item
        :param query: The query item
        :param k: The number of results to return
        :return: A list of the top-k most similar items
        """
        dataset_vectors = np.array(self.embeddings)
        if not dataset_is_normalized and normalize:
          dataset_vectors = self.normalize(dataset_vectors)
            
        query_normalized = self.normalize(query) if normalize else query

        # Compute cosine similarity between query and each item in the dataset
        scores = query_normalized @ dataset_vectors.T
        partitioned_indices = np.argpartition(-scores, kth=k)[:k]
        top_k_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]

        # Return the top-k most-similar items
        return [self.get_by_id(self.ids[i]) for i in top_k_indices]

class PCAIndex(AbstractIndex):
    def __init__(self, ids, embeddings, n_components):
        self.ids = ids
        self.pca = PCA(n_components=n_components)
        self.embeddings = self.pca.fit_transform(embeddings)

    def add(self, id, embedding):
        transformed_embedding = self.pca.fit_transform(embedding)
        self.ids.append(id)
        self.embeddings.append(transformed_embedding)

    def get_all(self):
        return {'id': self.ids, 'embedding': self.embeddings}

    def get_by_id(self, id):
        index = self.ids.index(id)
        return {'id': id, 'embedding': self.embeddings[index]}

    def normalize(self, vectors):
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Time complexity: O(n_vectors + k log k)
    def get_similarity(self, query, k, normalize=True, dataset_is_normalized=False, useGPU=False):
        """
        Find the top-k most similar items to a query item
        :param query: The query item
        :param k: The number of results to return
        :return: A list of the top-k most similar items
        """
        dataset_vectors = np.array(self.embeddings)
        if not dataset_is_normalized and normalize:
          dataset_vectors = self.normalize(dataset_vectors)
            
        query_normalized = self.normalize(query) if normalize else query

        # Compute cosine similarity between query and each item in the dataset
        scores = query_normalized @ dataset_vectors.T
        partitioned_indices = np.argpartition(-scores, kth=k)[:k]
        top_k_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]

        # Return the top-k most-similar items
        return [self.get_by_id(self.ids[i]) for i in top_k_indices]

class SQLiteDatabase:
    def __init__(self, filename):
        self.conn = sqlite3.connect(filename)
        self.conn.text_factory = bytes
        self.indexes = {}
        self.load_indexes()

    def insert(self, text, embedding):
        # Convert data to bytes using pickle
        embedding_as_bytes = pickle.dumps(embedding)

        try:
            cursor = self.conn.execute(
                "INSERT INTO main_table (text, embedding) VALUES (?, ?)",
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
                "SELECT * FROM main_table WHERE id = ?",
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
            cursor = self.conn.execute("SELECT * FROM main_table")
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
                "UPDATE main_table SET text = ?, embedding = ? WHERE id = ?",
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
                "DELETE FROM main_table WHERE id = ?",
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
    
    def load_indexes(self):
        # Retrieve the embeddings from the main table
        all_rows = self.get_all()
        ids = [row['id'] for row in all_rows]
        embeddings = [row['embedding'] for row in all_rows]

        # Retrieve index metadata from the database and instantiate indexes accordingly
        cursor = self.conn.execute("SELECT * FROM index_metadata")
        for row in cursor:
            index_name, index_type, index_params = row
            index_params = json.loads(index_params)
            if index_type == 'Base':
                self.indexes[index_name] = BaseIndex(embeddings, ids)
            elif index_type == 'PCA':
                self.indexes[index_name] = PCAIndex(embeddings, ids, index_params['n_components'])
            else:
                raise ValueError(f"Unknown index type: {index_type}")

    def create_index(self, index_name, index_type, index_params=None):
        # As before, but also create a record in the index_metadata table
        self.conn.execute(
            "INSERT INTO index_metadata (index_name, index_type, index_params) VALUES (?, ?, ?)",
            (index_name, index_type, json.dumps(index_params))
        )
        self.conn.commit()

    def refit_index(self, index_name):
        # Retrieve the index from the dictionary
        index = self.indexes[index_name]

        # Retrieve the embeddings from the main table
        all_rows = self.get_all()
        ids = [row['id'] for row in all_rows]
        embeddings = [row['embedding'] for row in all_rows]
        
        # Create a new index of the same type
        if isinstance(index, BaseIndex):
            new_index = BaseIndex(embeddings, ids)
        elif isinstance(index, PCAIndex):
            new_index = PCAIndex(embeddings, ids, index.n_components)
        else:
            raise ValueError(f"Unknown index type: {type(index)}")
        
        # Replace the index in the dictionary
        self.indexes[index_name] = new_index


    def query(self, index_name, query_embedding, k=10):
        # Retrieve the index from the dictionary
        index = self.indexes[index_name]

        # Perform the query
        results = index.get_similarity(query_embedding, k)

        # Return the results
        return results