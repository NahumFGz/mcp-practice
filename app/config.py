"""
Módulo de gestión de configuración de la aplicación.

Este módulo proporciona un sistema centralizado de configuración utilizando Pydantic settings.
Maneja las variables de entorno para conexiones a base de datos y servidores NATS,
proporcionando configuración con validación de tipos y mensajes de error útiles.

El módulo utiliza BaseSettings de Pydantic para la carga automática de variables
de entorno y validación, con soporte para configuración mediante archivo .env.
"""

import sys
from functools import lru_cache
from typing import List

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DB_NAME: str = Field(..., env="DB_NAME")
    DB_USERNAME: str = Field(..., env="DB_USERNAME")
    DB_PASSWORD: str = Field(..., env="DB_PASSWORD")
    DB_HOST: str = Field(..., env="DB_HOST")
    DB_PORT: int = Field(..., env="DB_PORT")

    NATS_SERVERS: str = Field(..., env="NATS_SERVERS")

    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    TAVILY_API_KEY: str = Field(..., env="TAVILY_API_KEY")

    @property
    def database_url(self) -> str:
        """Devuelve la URL async de conexión a PostgreSQL."""
        return (
            f"postgresql+asyncpg://{self.DB_USERNAME}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def database_memory_url(self) -> str:
        """Devuelve la URL async de conexión a PostgreSQL para la memoria."""
        return f"postgresql://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?sslmode=disable"

    @property
    def nats_servers_list(self) -> List[str]:
        """
        Devuelve la lista de servidores NATS como lista de strings.
        Se obtiene dividiendo NATS_SERVERS.
        """
        return [s.strip() for s in self.NATS_SERVERS.split(",") if s.strip()]

    class Config:
        """Configuración de Pydantic."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    """
    Instancia única de Settings.
    Si falta una variable, imprime error y termina el proceso.
    """
    try:
        return Settings()
    except ValidationError as e:
        print("❌ Invalid environment variables:", e, file=sys.stderr)
        sys.exit(1)
