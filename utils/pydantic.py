from functools import wraps

from pydantic import BaseModel


def pydantic_to_dict(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        result, status_code = f(*args, **kwargs)
        if isinstance(result, BaseModel):
            return result.model_dump(), status_code
        return result, status_code

    return decorated_function
