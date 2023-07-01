import numpy as np


def Normalize(datum):
    """
    Normalize a datum
    :param datum: The datum to normalize
    :return: The normalized datum
    """
    if datum.ndim == 1:
      return datum / np.linalg.norm(datum)
    
    if datum.ndim == 2:
      return datum / np.linalg.norm(datum, axis=1)[:, None]
    
    raise ValueError("Datum should be 1 or 2-dimensional")

# Time complexity: O(n_vectors + k log k)
def GetSimilarity(query, dataset, k, normalize=True, dataset_is_normalized=False, useGPU=False):
    """
    Find the top-k most similar items to a query item
    :param query: The query item
    :param dataset: The dataset to search (normalized)
    :param k: The number of results to return
    :return: A list of the top-k most similar items
    """
    dataset_vectors = np.array(dataset['embedding'].tolist())
    if not dataset_is_normalized and normalize:
      dataset_vectors = Normalize(dataset_vectors)
        
    query_normalized = Normalize(query) if normalize else query

    # Compute cosine similarity between query and each item in the dataset
    scores = query_normalized @ dataset_vectors.T
    partitioned_indices = np.argpartition(-scores, kth=k)[:k]
    top_k_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]

    # Return the top-k most-similar items
    return dataset[top_k_indices]

# Time complexity: O(n_vectors + k log k)
def GetSimilarityCPU(query, dataset, k, normalize=True, dataset_is_normalized=False, useGPU=False):
    """
    Find the top-k most similar items to a query item
    :param query: The query item
    :param dataset: The dataset to search (normalized)
    :param k: The number of results to return
    :return: A list of the top-k most similar items
    """
    dataset_vectors = np.array(dataset['embedding'].tolist())
    if not dataset_is_normalized and normalize:
      dataset_vectors = Normalize(dataset_vectors)
        
    query_normalized = Normalize(query) if normalize else query

    # Compute cosine similarity between query and each item in the dataset
    scores = query_normalized @ dataset_vectors.T
    partitioned_indices = np.argpartition(-scores, kth=k)[:k]
    top_k_indices = partitioned_indices[np.argsort(-scores[partitioned_indices])]

    # Return the top-k most-similar items
    return dataset[top_k_indices]



