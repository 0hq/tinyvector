from pydantic import BaseModel


class SuccessMessage(BaseModel):
    status: str = "success"


class ErrorMessage(BaseModel):
    error: str = ""
