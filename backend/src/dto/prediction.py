from typing import List
from pydantic import BaseModel


class CategoryDto(BaseModel):
    name: str
    probability: float


class PredictionCategoryDto(BaseModel):
    image_id: str
    categories: List[CategoryDto]
    level: int
