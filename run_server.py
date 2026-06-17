import os

import uvicorn

from app.common.logging_config import build_uvicorn_log_config, ensure_logging_configured, resolve_log_dir

if __name__ == "__main__":
    log_dir = resolve_log_dir()
    ensure_logging_configured()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("OCR_PORT", "8088")),
        log_config=build_uvicorn_log_config(log_dir),
        access_log=True,
        workers=1,
    )
