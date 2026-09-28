from fastapi import Request
from fastapi.responses import JSONResponse
from core.logging import get_logger, request_id_var
from config.settings import get_settings

logger = get_logger(__name__)

async def global_exception_handler(request: Request, exc: Exception):
    req_id = request_id_var.get()
    settings = get_settings()
    
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    error_detail = {
        "success": False,
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred on the server.",
            "request_id": req_id
        }
    }
    
    if settings.APP_ENV == "development":
        error_detail["error"]["details"] = str(exc)
        
    return JSONResponse(status_code=500, content=error_detail)
