import os
import tempfile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.common.logging_config import get_logger, resolve_log_dir
from app.common.ocr_engine import warmup_ocr
from app.common.pose_engine import warmup_pose_engine
from app.parsers.bankcard_parser import parse_bank_card
from app.parsers.idcard_parser import parse_idcard
from app.parsers.license_parser import parse_business_license
from app.parsers.portrait_validator import validate_portrait_image

ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
logger = get_logger("app.main")

app = FastAPI(title="NH OCR Service", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    log_dir = resolve_log_dir()
    logger.info("OCR service starting, log_dir=%s", log_dir)
    warmup_ocr()
    try:
        warmup_pose_engine()
    except Exception:
        logger.exception("portrait engine warmup skipped or failed")
    logger.info("OCR service ready, log_dir=%s", log_dir)


def _ok(data: Any) -> JSONResponse:
    return JSONResponse({"code": 200, "success": True, "data": data})


def _fail(message: str, status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        {"code": status_code, "success": False, "message": message, "data": None},
        status_code=status_code,
    )


def _validate_upload(upload: UploadFile) -> None:
    filename = (upload.filename or "").lower()
    suffix = os.path.splitext(filename)[1]
    if suffix and suffix not in ALLOWED_SUFFIX:
        raise HTTPException(status_code=400, detail=f"unsupported file type: {suffix or 'unknown'}")


async def _save_upload(upload: UploadFile) -> str:
    _validate_upload(upload)
    suffix = os.path.splitext(upload.filename or "")[1] or ".jpg"
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_file.close()
    content = await upload.read()
    if not content:
        os.remove(tmp_file.name)
        raise HTTPException(status_code=400, detail="empty file")
    with open(tmp_file.name, "wb") as handle:
        handle.write(content)
    return tmp_file.name


def _cleanup(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        os.remove(path)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "nh-ocr-service"}


@app.post("/ocr/bankcard")
async def ocr_bankcard(file: UploadFile = File(...)) -> JSONResponse:
    image_path = None
    try:
        image_path = await _save_upload(file)
        upload_name = file.filename or os.path.basename(image_path)
        logger.info("bankcard request file=%s", upload_name)
        result = parse_bank_card(image_path)
        logger.info(
            "bankcard response file=%s card_no=%s partial=%s bank=%s",
            upload_name,
            result.get("card_no"),
            result.get("card_no_partial"),
            result.get("bank_name"),
        )
        return _ok(result)
    except HTTPException as ex:
        return _fail(str(ex.detail), ex.status_code)
    except Exception:
        logger.exception("bankcard ocr failed")
        return _fail("bankcard ocr failed")
    finally:
        _cleanup(image_path)


def _portrait_enabled() -> bool:
    value = os.getenv("PORTRAIT_ENABLED", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


@app.post("/validate/portrait")
async def validate_portrait(file: UploadFile = File(...)) -> JSONResponse:
    if not _portrait_enabled():
        return _fail("portrait validation disabled", status_code=503)

    image_path = None
    try:
        image_path = await _save_upload(file)
        upload_name = file.filename or os.path.basename(image_path)
        logger.info("portrait validate request file=%s", upload_name)
        result = validate_portrait_image(image_path)
        logger.info(
            "portrait validate response file=%s passed=%s reason=%s",
            upload_name,
            result.get("passed"),
            result.get("reason"),
        )
        return _ok(result)
    except HTTPException as ex:
        return _fail(str(ex.detail), ex.status_code)
    except RuntimeError as ex:
        logger.exception("portrait validate unavailable")
        return _fail(str(ex), status_code=503)
    except Exception:
        logger.exception("portrait validate failed")
        return _fail("portrait validate failed")
    finally:
        _cleanup(image_path)


@app.post("/ocr/business-license")
async def ocr_business_license(file: UploadFile = File(...)) -> JSONResponse:
    image_path = None
    try:
        image_path = await _save_upload(file)
        upload_name = file.filename or os.path.basename(image_path)
        logger.info("business-license request file=%s", upload_name)
        result = parse_business_license(image_path)
        logger.info(
            "business-license response file=%s company_name=%s credit_code=%s",
            upload_name,
            result.get("company_name"),
            (result.get("credit_code") or "")[:4] + "****" if result.get("credit_code") else None,
        )
        return _ok(result)
    except HTTPException as ex:
        return _fail(str(ex.detail), ex.status_code)
    except Exception:
        logger.exception("business-license ocr failed")
        return _fail("business-license ocr failed")
    finally:
        _cleanup(image_path)


@app.post("/ocr/idcard")
async def ocr_idcard(
    file: UploadFile = File(...),
    side: str = Query(default="auto", pattern="^(auto|front|back)$"),
) -> JSONResponse:
    image_path = None
    try:
        image_path = await _save_upload(file)
        upload_name = file.filename or os.path.basename(image_path)
        logger.info("idcard request file=%s side=%s", upload_name, side)
        result = parse_idcard(image_path, side=side)
        id_no = result.get("id_no")
        masked_id = f"{id_no[:6]}****{id_no[-4:]}" if id_no and len(id_no) >= 8 else id_no
        logger.info(
            "idcard response file=%s side=%s id_no=%s name=%s",
            upload_name,
            result.get("side"),
            masked_id,
            result.get("name"),
        )
        return _ok(result)
    except HTTPException as ex:
        return _fail(str(ex.detail), ex.status_code)
    except Exception:
        logger.exception("idcard ocr failed")
        return _fail("idcard ocr failed")
    finally:
        _cleanup(image_path)


if __name__ == "__main__":
    import uvicorn

    from app.common.logging_config import build_uvicorn_log_config, resolve_log_dir

    log_dir = resolve_log_dir()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("OCR_PORT", "8088")),
        log_config=build_uvicorn_log_config(log_dir),
        access_log=True,
        reload=False,
        workers=1,
    )
