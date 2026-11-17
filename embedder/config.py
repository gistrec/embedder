from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="EMBEDDER_", extra="ignore")

    # Модель fastembed. Смена модели = другие векторы: коллекции клиентов переиндексировать.
    model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Куда fastembed кладёт веса. Внутри каталога приложения, чтобы переживать рестарты.
    cache_dir: str = "models"
    # Потоки ONNX Runtime. Хост общий: не отдаём модели все ядра.
    threads: int = 2
    # Bearer-токен. Пусто — без авторизации (только для локальной разработки).
    token: str = ""
    max_inputs: int = 64
    # Модель читает только первые 128 токенов (~500–1000 символов), остальное молча
    # отбрасывает. Лимит близок к этому окну, чтобы длинный текст получал 413, а не обрезался.
    max_chars: int = 1000


settings = Settings()
