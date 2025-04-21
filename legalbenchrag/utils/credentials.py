import tomllib
from pydantic import BaseModel, SecretStr


class AICredentials(BaseModel):
    openai_api_key: SecretStr
    anthropic_api_key: SecretStr
    cohere_api_key: SecretStr
    voyageai_api_key: SecretStr


class DropboxCredentials(BaseModel):
    token: SecretStr


class Credentials(BaseModel):
    ai: AICredentials
    dropbox: DropboxCredentials


with open("./credentials/credentials.toml", "rb") as credentials_file:
    credentials = Credentials.model_validate(tomllib.load(credentials_file))
