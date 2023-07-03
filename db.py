import sqlite3
import pickle
import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
import uuid

def norm(datum):
    axis = None if datum.ndim == 1 else 1
    return datum / np.linalg.norm(datum, axis=axis, keepdims=True)

class AbstractIndex(ABC):
    def __init__(self, table_name, type, num_vectors, dimension, allow_updates=False):
        self.table_name = table_name
        self.type = type
        self.num_vectors = num_vectors
        self.dimension = dimension
        self.allow_updates = allow_updates

    def __len__(self):
        return self.num_vectors

    def __str__(self):
        return self.table_name + ' ' + self.type

    @abstractmethod
    def add_vector(self, id, embedding):
        pass

    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def get_by_id(self, id):
        pass

    @abstractmethod
    def get_similarity(self, query, k):
        pass


class BaseIndex(AbstractIndex):
    def __init__(self, embeddings, ids, normalize, allow_updates=True):
        super().__init__('base', len(embeddings), len(embeddings[0]), allow_updates)  # Initialize name/len attribute in parent class
        self.embeddings = norm(embeddings) if normalize else embeddings
        self.ids = ids
        self.normalize = normalize
        
    def add_vector(self, id, embedding):
        if not self.allow_updates:
            raise ValueError("Cannot add vectors to a non-updatable index")
        if len(embedding) != self.dimension:
            raise ValueError(f"Expected embedding of dimension {self.dimension}, got {len(embedding)}")
        self.ids.append(id)
        self.embeddings.append(norm(embedding) if self.normalize else embedding)
        self.num_vectors += 1

    def get_all(self):
        return {'id': self.ids, 'embedding': self.embeddings}

    def get_by_id(self, id):
        index = self.ids.index(id)
        return {'id': id, 'embedding': self.embeddings[index]}

    # Time complexity: O(n_vectors + k log k)
    def get_similarity(self, query, k):
        # Check dimensions of query
        if len(query) != self.dimension:
            raise ValueError(f"Expected query of dimension {self.dimension}, got {len(query)}")
        
        # Normalize query 
        dataset_vectors = np.array(self.embeddings)
        query_normalized = norm(query) if self.normalize else query

        # Compute cosine similarity between query and each item in the dataset
        scores = query_normalized @ dataset_vectors.T
        partitioned_indices = np.argpartition(-scores, kth=k)[:k]
        top_k_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]

        # Return the top-k most-similar items
        return [self.get_by_id(self.ids[i]) for i in top_k_indices]



class PCAIndex(AbstractIndex):
    def __init__(self, ids, embeddings, n_components, normalize, allow_updates=False):
        if allow_updates:
            print("Warning: PCA index is not intended to be used with allow_updates=True. This will result in reduced performance until the next time index is rebuilt.")
        super().__init__('pca', len(embeddings), n_components, allow_updates)  # Initialize name/len attribute in parent class
        self.ids = ids
        self.pca = PCA(n_components)
        self.original_dimension = len(embeddings[0])
        self.embeddings = self.pca.fit_transform(embeddings)
        self.embeddings = norm(self.embeddings) if normalize else self.embeddings
        self.normalize = normalize

    def add_vector(self, id, embedding):
        if not self.allow_updates:
            raise ValueError("Cannot add vectors to a non-updatable index")
        print("Warning: PCA index is not intended to be used with allow_updates=True. This will result in reduced performance until the next time index is rebuilt.")
        if len(embedding) != self.original_dimension:
            raise ValueError(f"Expected embedding of dimension {self.original_dimension}, got {len(embedding)}")
        self.ids.append(id)
        transformed_embedding = self.pca.transform(embedding.reshape(1, -1))
        self.embeddings.append(norm(transformed_embedding) if self.normalize else transformed_embedding)
        self.num_vectors += 1

    def get_all(self):
        return {'id': self.ids, 'embedding': self.embeddings}

    def get_by_id(self, id):
        index = self.ids.index(id)
        return {'id': id, 'embedding': self.embeddings[index]}

    # Time complexity: O(n_vectors + k log k)
    def get_similarity(self, query, k):
        # Check dimensions of query
        if len(query) != self.original_dimension:
            raise ValueError(f"Expected query of dimension {self.original_dimension}, got {len(query)}")
        
        # Transform query to PCA space
        dataset_vectors = np.array(self.embeddings)   
        transformed_query = self.pca.transform(query.reshape(1, -1))
        query_normalized = norm(transformed_query) if self.normalize else transformed_query

        # Compute cosine similarity between query and each item in the dataset
        scores = query_normalized @ dataset_vectors.T
        partitioned_indices = np.argpartition(-scores, kth=k)[:k]
        top_k_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]

        # Return the top-k most-similar items
        return [self.get_by_id(self.ids[i]) for i in top_k_indices]

class SQLiteDatabase:
    def __init__(self, filename, dimensions, load_indexes=True):
        self.filename = filename
        self.dimensions = dimensions
        self.conn = sqlite3.connect(filename)
        self.conn.text_factory = bytes

        # Create the table if it doesn't exist
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS main_table (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, embedding BLOB NOT NULL)"
        )
        # Create index table if it doesn't exist
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS index_metadata (name TEXT PRIMARY KEY, type TEXT NOT NULL, num_vectors INTEGER NOT NULL, is_active BOOLEAN NOT NULL, normalize BOOLEAN NOT NULL, dimension INTEGER NOT NULL)")
        self.conn.commit()

        # Create indexes
        self.indexes = {}
        if load_indexes:
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
    
    def insert_many(self, texts, embeddings):
        data = [(text, pickle.dumps(embedding)) for text, embedding in zip(texts, embeddings)]
        try:
            cursor = self.conn.executemany(
                "INSERT INTO main_table (text, embedding) VALUES (?, ?)",
                data
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Insertion error: {e}")
            return None

        return cursor.lastrowid

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
        try:
            # Retrieve the embeddings from the main table
            all_rows = self.get_all()
            ids = [row['id'] for row in all_rows]
            embeddings = [row['embedding'] for row in all_rows]

            # Retrieve index metadata from the database and instantiate indexes accordingly
            cursor = self.conn.execute("SELECT * FROM index_metadata")
            for index_row in cursor:
                index_name, index_type, _, is_active, normalize, dimension = index_row
                if not is_active:
                    continue
                if index_type == 'Base':
                    self.indexes[index_name] = BaseIndex(embeddings, ids, normalize)
                elif index_type == 'PCA':
                    self.indexes[index_name] = PCAIndex(embeddings, ids, dimension, normalize)
                else:
                    raise ValueError(f"Unknown index type: {index_type}")
        except MemoryError:
            print("Failed to load all indexes due to insufficient memory. Please deactivate some indexes and try again.")
            return None

    def create_index(self, index_type, index_name=None, dimensions=None, cosine_similarity=True):
        if index_type not in ['Base', 'PCA']:
            print("Unknown index type.")
            return None
        if index_name is None:
            index_name = index_type + '_' + str(uuid.uuid4())
        if index_type == 'PCA' and dimensions is None:
            print("Please specify the dimension for the index.")
            return None
        if dimensions is None:
            dimensions = self.dimensions

        # Check if the index already exists
        if self.index_exists(index_name):
            print(f"Index {index_name} already exists.")
            return None
        
        # Insert index metadata into the database
        self.conn.execute(
            "INSERT INTO index_metadata (index_name, index_type, num_vectors, is_active, normalize, dimension) VALUES (?, ?, ?, ?, ?, ?)",
            (index_name, index_type, len(self), True, cosine_similarity, dimensions)
        )
        self.conn.commit()

        # Create the index
        self.rebuild_index(index_name)

    def rebuild_index(self, index_name):
        # Retrieve the index from the database
        cursor = self.conn.execute(
            "SELECT * FROM index_metadata WHERE index_name = ?",
            (index_name,)
        )
        index_row = cursor.fetchone()
        if index_row is None:
            print(f"Index {index_name} does not exist.")
            return None
        
        index_name, index_type, _, is_active, normalize, dimension = index_row

        if not is_active:
            print(f"Index {index_name} is not active.")
            return None

        # Retrieve the embeddings from the main table
        all_rows = self.get_all()
        ids = [row['id'] for row in all_rows]
        embeddings = [row['embedding'] for row in all_rows]
        
        # Create a new index of the same type
        if index_type == 'Base':
            new_index = BaseIndex(embeddings, ids, normalize)
        elif index_type == 'PCA':
            new_index = PCAIndex(embeddings, ids, dimension, normalize)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Replace the index in the dictionary
        self.indexes[index_name] = new_index

    def delete_index(self, index_name):
        # As before, but also delete the record in the index_metadata table
        self.conn.execute(
            "DELETE FROM index_metadata WHERE index_name = ?",
            (index_name,)
        )
        self.conn.commit()

        # Delete the index from the dictionary
        del self.indexes[index_name]

    def index_exists(self, index_name):
        # Retrieve the index from the database
        cursor = self.conn.execute(
            "SELECT * FROM index_metadata WHERE index_name = ?",
            (index_name,)
        )
        index_row = cursor.fetchone()
        return index_row is not None

    def set_index_activity(self, index_name, is_active):
        if not self.index_exists(index_name):
            print(f"Index {index_name} does not exist.")
            return None

        # As before, but also update the record in the index_metadata table
        self.conn.execute(
            "UPDATE index_metadata SET is_active = ? WHERE index_name = ?",
            (is_active, index_name)
        )
        self.conn.commit()

        # Update the index
        self.rebuild_index(index_name)

    def query(self, index_name, query_embedding, k=10):
        # Retrieve the index from the dictionary
        index = self.indexes[index_name]

        # Perform the query
        results = index.get_similarity(query_embedding, k)

        # Return the results
        return results