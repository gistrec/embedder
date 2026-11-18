"""Модель эмбеддингов: загрузка, префиксы, вычисление.

Часть моделей обучена с префиксами: e5 ждёт ``query: `` у вопроса и ``passage: `` у документа,
без них качество поиска заметно падает. Клиенты передают только ``type``, а префикс
подставляет сервис — так смена модели не требует правок у клиентов.
"""
import logging
from dataclasses import dataclass

import numpy as np

from embedder.config import settings

logger = logging.getLogger("embedder")

# Имя модели в нижнем регистре -> {type -> префикс}. Модели без записи работают без
# префиксов (симметричные). fastembed принимает имя в любом регистре, поэтому и поиск
# здесь регистронезависимый.
_NOMIC = {"query": "search_query: ", "document": "search_document: "}
PREFIXES: dict[str, dict[str, str]] = {
    "intfloat/multilingual-e5-large": {"query": "query: ", "document": "passage: "},
    "nomic-ai/nomic-embed-text-v1": _NOMIC,
    "nomic-ai/nomic-embed-text-v1.5": _NOMIC,
    "nomic-ai/nomic-embed-text-v1.5-q": _NOMIC,
}
_PREFIX_NOTE = "Prefixes for queries/documents:"


def prefixes_for(name: str) -> dict[str, str]:
    return PREFIXES.get(name.lower(), {})


def missing_prefixes(name: str, description: str) -> bool:
    """Модель по описанию fastembed требует префиксов, а в PREFIXES её нет."""
    _, found, note = description.partition(_PREFIX_NOTE)
    return bool(found) and not note.strip().lower().startswith("not") and not prefixes_for(name)


@dataclass
class Embedder:
    name: str
    dim: int
    _model: object

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        prefix = prefixes_for(self.name).get(kind, "")
        inputs = [prefix + t for t in texts]
        vectors = np.asarray(list(self._model.embed(inputs, batch_size=32)), dtype=np.float32)
        # fastembed отдаёт эту модель без нормализации (норма ~3,5). Приводим к единичной
        # длине, чтобы dot product и cosine у клиентов давали одинаковый порядок.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.where(norms == 0, 1, norms)).tolist()


def load(name: str | None = None) -> Embedder:
    from fastembed import TextEmbedding

    name = name or settings.model
    model = TextEmbedding(name, cache_dir=settings.cache_dir, threads=settings.threads)
    for known in TextEmbedding.list_supported_models():
        if known["model"].lower() == name.lower() and missing_prefixes(name, known.get("description", "")):
            logger.warning(
                "model %s требует префиксов query/document, но в PREFIXES её нет: "
                "качество поиска будет ниже. Добавьте запись в embedder/model.py",
                name,
            )
    # Размерность берём из реального вектора, а не из справочника: так не разъедется.
    dim = len(next(iter(model.embed(["probe"]))))
    return Embedder(name=name, dim=dim, _model=model)


if __name__ == "__main__":
    # `python -m embedder.model` — скачать веса заранее (деплой, первый запуск).
    e = load()
    print(f"{e.name}: dim={e.dim}, cache={settings.cache_dir}")
