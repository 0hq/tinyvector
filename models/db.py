from typing import Optional
from pydantic import BaseModel


class TableCreationBody(BaseModel):
    table_name: str
    dimension: int
    use_uuid: bool = False


class TableDeletionBody(BaseModel):
    table_name: str


class ItemInsertionBody(BaseModel):
    table_name: str
    id: Optional[str] = None
    embedding: list[int]
    content: Optional[str] = None
    defer_index_update: bool = False
