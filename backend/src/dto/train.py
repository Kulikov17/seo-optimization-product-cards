from pydantic import BaseModel


class TrainStatusDto(BaseModel):
    task_id: str
    status: str
