// pm2-конфиг. Адрес и порт берём из .env (EMBEDDER_HOST / EMBEDDER_PORT): на сервере
// это IP в WireGuard-меше, чтобы сервис не торчал в интернет.
// Перед стартом: python -m venv .venv && .venv/bin/pip install -r requirements.txt
//               && .venv/bin/python -m embedder.model   (скачать веса заранее)
const fs = require("fs");
const path = require("path");

function dotenv(name, fallback) {
  try {
    const raw = fs.readFileSync(path.join(__dirname, ".env"), "utf8");
    const m = raw.match(new RegExp(`^${name}=(.*)$`, "m"));
    return m ? m[1].trim().replace(/^["']|["']$/g, "") : fallback;
  } catch {
    return fallback;
  }
}

module.exports = {
  apps: [
    {
      name: "embedder",
      cwd: __dirname,
      script: ".venv/bin/uvicorn",
      interpreter: "none",
      // Один воркер: каждый держит свою копию модели (~0,5 ГБ).
      args: `embedder.app:app --host ${dotenv("EMBEDDER_HOST", "127.0.0.1")} --port ${dotenv("EMBEDDER_PORT", "9110")} --workers 1`,
      autorestart: true,
      // Обычный пик модели по умолчанию — ~0,9 ГБ. Выше лимита считаем утечкой и
      // перезапускаем: хост общий. Для модели крупнее поднимите EMBEDDER_MAX_MEMORY,
      // иначе pm2 будет перезапускать здоровый процесс.
      max_memory_restart: dotenv("EMBEDDER_MAX_MEMORY", "1500M"),
      exp_backoff_restart_delay: 100,
      max_restarts: Number.MAX_SAFE_INTEGER,
    },
  ],
};
