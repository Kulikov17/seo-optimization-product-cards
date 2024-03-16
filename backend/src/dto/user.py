from pydantic import BaseModel


class UserDto(BaseModel):
    chat_id: str
    username: str
    first_name:  str | None = None
    last_name: str | None = None
