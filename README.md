# embedder

Маленький HTTP-сервис эмбеддингов на [fastembed](https://github.com/qdrant/fastembed)
(ONNX Runtime, без torch). Один процесс держит модель в памяти и отдаёт векторы нескольким
проектам, чтобы каждый не грузил свою копию.

```bash
curl -s http://127.0.0.1:9110/v1/embed \
  -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' \
  -d '{"input": ["почему не открывается кабинет клиента"], "type": "query"}'
# {"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "dim": 384, "data": [[...]]}
```

## API

| Метод | Тело | Ответ |
|---|---|---|
| `POST /v1/embed` | `{"input": [строки], "type": "query" \| "document"}` | `{"model", "dim", "data": [[float]]}` |
| `GET /health` | — | `{"status": "ok", "model", "dim"}` (без токена) |

- `type` — что кодируем: вопрос (`query`) или фрагмент корпуса (`document`, по умолчанию).
  Моделям с префиксами (e5: `query: ` / `passage: `) сервис подставляет их сам, клиенту
  о них знать не нужно.
- Лимиты: до 64 строк за запрос, до 1000 символов в строке (`413`), пустые строки — `422`.
- Модель по умолчанию читает только первые 128 токенов текста (~500–1000 символов),
  остальное отбрасывает. Длинные документы режьте на фрагменты по ~500 символов на
  стороне клиента.
- Векторы нормализованы до единичной длины: cosine и dot product дают один порядок.
- Клиенту стоит сверять `model` и `dim` из ответа с тем, с чем индексировалась его
  коллекция: векторы разных моделей несравнимы.

## Ресурсы

Модель по умолчанию — `paraphrase-multilingual-MiniLM-L12-v2` (384 dim, ~240 МБ весов,
хорошо работает с русским). Замер: процесс держит ~0,5 ГБ RAM постоянно (не только во
время запросов), запрос из одной строки — ~10 мс на CPU, 200 длинных фрагментов — ~13 с.
Запросы обрабатываются по одному, ONNX Runtime использует `EMBEDDER_THREADS` ядер.

## Запуск

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -m embedder.model      # скачать веса в ./models (≈2–3 мин в первый раз)
pm2 start ecosystem.config.js           # или: .venv/bin/uvicorn embedder.app:app --port 9110
```

Настройки — переменные `EMBEDDER_*` в `.env` (см. `.env.example`). На сервере
`EMBEDDER_HOST` — адрес во внутренней сети (WireGuard), а не `0.0.0.0`.

## Тесты

```bash
.venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest
```

## Лицензия

MIT
