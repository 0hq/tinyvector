import pytest
from pydantic import ValidationError

from models.db import TableMetadata


def test_invalid_table_metadata():
    """
    We raise an error when allow_index_updates is True and index type is PCA since we don't support this functionality.
    """
    with pytest.raises(ValidationError) as exc_info:
        TableMetadata(
            table_name="example",
            allow_index_updates=True,
            dimension=3,
            index_type="pca",
            is_index_active=True,
            normalize=True,
            use_uuid=1,
        )
