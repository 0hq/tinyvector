from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel


class TableCreationBody(BaseModel):
    table_name: str
    dimension: int
    use_uuid: bool = False


class TableDeletionBody(BaseModel):
    table_name: str


class IndexType(str, Enum):
    BRUTE_FORCE = "brute_force"
    PCA = "pca"


class IndexDeletionBody(BaseModel):
    table_name: str


class IndexCreationBody(BaseModel):
    table_name: str
    index_type: IndexType
    normalize: bool = True
    allow_index_updates: bool = False
    n_components: Optional[int] = None


class ItemInsertionBody(BaseModel):
    table_name: str
    id: Optional[str] = None
    embedding: list[int]
    content: Optional[str] = None
    defer_index_update: bool = False


class TableQueryObject(BaseModel):
    table_name: str
    query: list[int]
    k: int


class TableQueryResultInstance(BaseModel):
    content: str
    embedding: list[int]
    id: str
    score: float


class TableQueryResult(BaseModel):
    items: list[TableQueryResultInstance]


class TableMetadata(BaseModel):
    table_name: str
    allow_index_updates: bool
    dimension: int
    index_type: IndexType
    is_index_active: bool
    normalize: bool
    use_uuid: int


class TableName(str):
    pass


class DatabaseInfo(BaseModel):
    indexes: list[str]
    num_indexes: int
    num_tables: int
    tables: Dict[TableName, TableMetadata]
