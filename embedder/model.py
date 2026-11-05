"""Модель эмбеддингов: загрузка, префиксы, вычисление.

Часть моделей обучена с префиксами: e5 ждёт ``query: `` у вопроса и ``passage: `` у документа,
без них качество поиска заметно падает. Клиенты передают только ``type``, а префикс
подставляет сервис — так смена модели не требует правок у клиентов.
"""
from dataclasses import dataclass

from embedder.config import settings

# type -> префикс. Модели без записи работают без префиксов (симметричные).
PREFIXES: dict[str, dict[str, str]] = {
    "intfloat/multilingual-e5-large": {"query": "query: ", "document": "passage: "},
}


@dataclass
class Embedder:
    name: str
    dim: int
    _model: object

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        prefix = PREFIXES.get(self.name, {}).get(kind, "")
        inputs = [prefix + t for t in texts]
        return [vec.tolist() for vec in self._model.embed(inputs, batch_size=32)]


def load(name: str | None = None) -> Embedder:
    from fastembed import TextEmbedding

    name = name or settings.model
    model = TextEmbedding(name, cache_dir=settings.cache_dir, threads=settings.threads)
    # Размерность берём из реального вектора, а не из справочника: так не разъедется.
    dim = len(next(iter(model.embed(["probe"]))))
    return Embedder(name=name, dim=dim, _model=model)


if __name__ == "__main__":
    # `python -m embedder.model` — скачать веса заранее (деплой, первый запуск).
    e = load()
    print(f"{e.name}: dim={e.dim}, cache={settings.cache_dir}")
