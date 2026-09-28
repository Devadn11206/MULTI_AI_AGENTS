import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from core.logging import request_id_var

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        req_id = f"req_{uuid.uuid4().hex[:12]}"
        token = request_id_var.set(req_id)
        
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            request_id_var.reset(token)
