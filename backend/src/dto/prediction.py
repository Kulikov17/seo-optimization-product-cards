from pydantic import BaseModel


class PredictionDto(BaseModel):
    result: str
