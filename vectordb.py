import sqlite3
import numpy as np
from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
import psutil
import uuid

def norm_np(datum):
    axis = None if datum.ndim == 1 else 1
    return datum / np.linalg.norm(datum, axis=axis, keepdims=True)

def norm_python(datum):
    datum = np.array(datum)
    result = norm_np(datum)
    return result.tolist()


def array_to_blob(array):
    return array.tobytes()

def blob_to_array(blob):
    return np.frombuffer(blob, dtype=np.float32)

class AbstractIndex(ABC):
    def __init__(self, table_name, type, num_vectors, dimension, allow_updates=False):
        self.table_name = table_name
        self.type = type
        self.num_vectors = num_vectors
        self.dimension = dimension

    def __len__(self):
        return self.num_vectors

    def __str__(self):
        return self.table_name + ' ' + self.type

    @abstractmethod
    def add_vector(self, id, embedding):
        pass

    @abstractmethod
    def get_similarity(self, query, k):
        pass


class BruteForceIndexImmutable(AbstractIndex):
    def __init__(self, table_name, dimension, normalize, embeddings, ids):
        super().__init__(table_name, 'brute_force', len(embeddings), dimension)  
        self.embeddings = norm_np(embeddings) if normalize else embeddings
        self.ids = ids
        self.normalize = normalize
        
    def add_vector(self, id, embedding):
        raise NotImplementedError("This index does not support vector addition. Please use BruteForceIndexMutable if updates are required.")
        
    def get_similarity(self, query, k):
        if len(query) != self.dimension:
            raise ValueError(f"Expected query of dimension {self.dimension}, got {len(query)}")

        query_normalized = norm_np(query) if self.normalize else query
        scores = query_normalized @ self.embeddings.T
        partitioned_indices = np.argpartition(-scores, kth=k)[:k]
        top_k_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]

        return [{'id': self.ids[i], 'embedding': self.embeddings[i], 'score': scores[i]} for i in top_k_indices]


class BruteForceIndexMutable(AbstractIndex):
    def __init__(self, table_name, dimension, normalize, embeddings, ids):
        super().__init__(table_name, 'brute_force', len(embeddings), dimension) 
        self.embeddings = norm_python(embeddings) if normalize else embeddings.tolist()
        self.ids = ids
        self.normalize = normalize
        
    def add_vector(self, id, embedding):
        if len(embedding) != self.dimension:
            raise ValueError(f"Expected embedding of dimension {self.dimension}, got {len(embedding)}")
        self.ids.append(id)
        self.embeddings.append(norm_python(embedding) if self.normalize else embedding.tolist())
        self.num_vectors += 1
        
    def get_similarity(self, query, k):
        if len(query) != self.dimension:
            raise ValueError(f"Expected query of dimension {self.dimension}, got {len(query)}")

        query_normalized = norm_np(query) if self.normalize else query
        dataset_vectors = np.array(self.embeddings)
        scores = query_normalized @ dataset_vectors.T
        partitioned_indices = np.argpartition(-scores, kth=k)[:k]
        top_k_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]

        return [{'id': self.ids[i], 'embedding': self.embeddings[i], 'score': scores[i]} for i in top_k_indices]



class PCAIndex(AbstractIndex):
    def __init__(self, table_name, dimension, n_components, normalize, embeddings, ids):
        super().__init__(table_name, 'pca', len(embeddings), n_components)  # Initialize name/len attribute in parent class
        self.ids = ids
        self.pca = PCA(n_components)
        self.original_dimension = dimension
        self.embeddings = self.pca.fit_transform(embeddings)
        self.embeddings = norm_np(self.embeddings) if normalize else self.embeddings
        self.normalize = normalize

    def add_vector(self, id, embedding):
        raise NotImplementedError("This index does not support vector addition. Please use BruteForceIndexMutable if updates are required.")

    # Time complexity: O(n_vectors + k log k)
    def get_similarity(self, query, k):
        # Check dimensions of query
        if len(query) != self.original_dimension:
            raise ValueError(f"Expected query of dimension {self.original_dimension}, got {len(query)}")
        
        # Transform query to PCA spac
        dataset_vectors = self.embeddings
        transformed_query = self.pca.transform(query)
        query_normalized = norm_np(transformed_query) if self.normalize else transformed_query

        # Compute cosine similarity between query and each item in the dataset
        scores = query_normalized @ dataset_vectors.T
        partitioned_indices = np.argpartition(-scores, kth=k)[:k]
        top_k_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]

        # Return the top-k most-similar items
        return [{ 'id': self.ids[i], 'embedding': self.embeddings[i], 'score': scores[i] } for i in top_k_indices]


class DB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.c = self.conn.cursor()
        self.table_metadata = {}
        self.indexes = {}
        self._init_db()

    def _init_db(self):
        self.c.execute("CREATE TABLE IF NOT EXISTS table_metadata (table_name TEXT PRIMARY KEY, dimension INTEGER, index_type TEXT, normalize BOOLEAN, allow_index_updates BOOLEAN, is_active BOOLEAN, use_uuid BOOLEAN)")
        self.conn.commit()
        self._load_metadata()

    def _load_metadata(self):
        self.table_metadata = {}
        select_query = "SELECT * FROM table_metadata"
        for row in self.c.execute(select_query):
            table_name, dimension, index_type, normalize, allow_index_updates, is_index_active, use_uuid = row
            self.table_metadata[table_name] = {'dimension': dimension, 'index_type': index_type, 'normalize': normalize, 'allow_index_updates': allow_index_updates, 'is_index_active': is_index_active, 'use_uuid': use_uuid}
            if is_index_active:
                self.create_index(table_name, index_type, dimension, normalize, allow_index_updates)

    def create_index(self, table_name, index_type, normalize, allow_index_updates=None, n_components=None):
        if psutil.virtual_memory().available < 0.1 * psutil.virtual_memory().total:
            raise MemoryError("System is running out of memory")
        def get_data(select_query):
            self.c.execute(select_query)
            rows = self.c.fetchall()
            ids = [row[0] for row in rows]
            embeddings = [blob_to_array(row[1]) for row in rows]
            return ids, embeddings
        
        if self.table_metadata.get(table_name) is None:
            raise ValueError(f"Table {table_name} does not exist. Create the table first.")
        
        if self.indexes.get(table_name) is not None:
            raise ValueError(f"Index for table {table_name} already exists. Delete the index first if you want to rebuild it.")
        
        if allow_index_updates is None:
            allow_index_updates = True if index_type == 'brute_force' else False
        if allow_index_updates is True and index_type == 'pca':
            raise ValueError("PCA index does not support updates. Please set allow_index_updates=False.")
        
        dimension = self.table_metadata[table_name]['dimension']
        if index_type == 'brute_force':
            ids, embeddings = get_data(f"SELECT * FROM {table_name}")
            if allow_index_updates:
                self.indexes[table_name] = BruteForceIndexMutable(table_name, dimension, normalize, embeddings, ids)
            else:
                self.indexes[table_name] = BruteForceIndexImmutable(table_name, dimension, normalize, embeddings, ids)
        elif index_type == 'pca':
            if n_components is None:
                raise ValueError("n_components must be specified for PCA index")
            ids, embeddings = get_data(f"SELECT * FROM {table_name}") # In the future, we can just load the cached PCA vectors directly from the DB
            self.indexes[table_name] = PCAIndex(table_name, dimension, n_components, normalize, embeddings, ids)
        else:
            raise ValueError(f"Unknown index type {index_type}")
        
        # Update metadata
        self.table_metadata[table_name]['index_type'] = index_type
        self.table_metadata[table_name]['is_index_active'] = True
        self.table_metadata[table_name]['allow_index_updates'] = allow_index_updates
        self.table_metadata[table_name]['normalize'] = normalize
        self.c.execute("UPDATE table_metadata SET index_type = ?, is_active = ?, allow_index_updates = ?, normalize = ? WHERE table_name = ?", (index_type, True, allow_index_updates, normalize, table_name))
        self.conn.commit()


    def create_table(self, table_name, dimension, use_uuid):
        if self.table_metadata.get(table_name) is not None:
            raise ValueError(f"Table {table_name} already exists")
        self.c.execute(f"CREATE TABLE {table_name} (id TEXT PRIMARY KEY, embedding BLOB, content TEXT)")
        self.c.execute("INSERT INTO table_metadata VALUES (?, ?, ?, ?, ?, ?, ?)", (table_name, dimension, None, None, None, False, use_uuid))
        self.conn.commit()

        # Update metadata
        self.table_metadata[table_name] = {'dimension': dimension, 'index_type': None, 'normalize': None, 'allow_index_updates': None, 'is_index_active': False, 'use_uuid': use_uuid}

    def delete_table(self, table_name):
        if self.table_metadata.get(table_name) is None:
            raise ValueError(f"Table {table_name} does not exist")
        self.c.execute(f"DROP TABLE {table_name}")
        self.c.execute("DELETE FROM table_metadata WHERE table_name = ?", (table_name,))
        self.conn.commit()
        self.table_metadata.pop(table_name)
        
    def delete_index(self, table_name):
        if self.indexes.get(table_name) is None:
            raise ValueError(f"Index for table {table_name} does not exist")
        self.indexes[table_name].delete()
        self.indexes.pop(table_name)
        self.table_metadata[table_name]['is_index_active'] = False
        self.table_metadata[table_name]['index_type'] = None
        self.table_metadata[table_name]['normalize'] = None
        self.table_metadata[table_name]['allow_index_updates'] = None
        self.c.execute("UPDATE table_metadata SET index_type = NULL, normalize = NULL, allow_index_updates = NULL, is_active = 0 WHERE table_name = ?", (table_name,))
        self.conn.commit()

    def insert(self, table_name, id, embedding, content, defer_index_update=False):
        if psutil.virtual_memory().available < 0.1 * psutil.virtual_memory().total:
            raise MemoryError("System is running out of memory")

        if self.table_metadata[table_name]['use_uuid'] and id is not None:
            raise ValueError("This table uses auto-generated UUIDs. Do not provide an ID.")
        elif self.table_metadata[table_name]['use_uuid']:
            id = str(uuid.uuid4())  # Generate a unique ID using the uuid library.
        elif id is None:
            raise ValueError("This table uses custom IDs. Please provide an ID.")

        insert_query = f"INSERT INTO {table_name} VALUES (?, ?, ?)"
        self.c.execute(insert_query, (id, array_to_blob(embedding), content))
        self.conn.commit()

        if self.table_metadata[table_name]['is_index_active'] is True and self.table_metadata[table_name]['allow_index_updates'] is True and defer_index_update is False:
            self.indexes[table_name].add_vector(id, embedding)

    def query(self, table_name, query, k):
        if self.table_metadata.get(table_name) is None:
            raise ValueError(f"Table {table_name} does not exist")
        if self.table_metadata[table_name]['is_index_active'] is False:
            raise ValueError(f"Index for table {table_name} does not exist")
        
        items = self.indexes[table_name].get_similarity(query, k)

        # Get content from DB in a single query
        ids = [item['id'] for item in items]
        placeholder = ", ".join(["?"] * len(ids))  # Generate placeholders for query
        self.c.execute(f"SELECT id, content FROM {table_name} WHERE id IN ({placeholder})", ids)
        content_dict = {id: content for id, content in self.c.fetchall()}

        # Add the content to the items
        for item in items:
            item['content'] = content_dict.get(item['id'])

        return items