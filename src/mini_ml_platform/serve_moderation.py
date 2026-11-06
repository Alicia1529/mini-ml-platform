import asyncio
import json
import os

from fastapi.responses import JSONResponse
from ray import serve
from starlette.requests import Request

from mini_ml_platform.moderation import Message, moderate_message


@serve.deployment(
    max_ongoing_requests=1,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "target_ongoing_requests": 0.5,
        "upscale_delay_s": 1,
        "downscale_delay_s": 10,
    },
)
class ModerationService:
    async def __call__(self, request: Request):
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            return JSONResponse({"error": "text must be a string"}, status_code=400)

        await asyncio.sleep(0.2)
        text = data["text"]
        message = Message(id=0, text=text)
        result = moderate_message(message)
        return {
            "flagged": result.flagged,
            "matched_keywords": list(result.matched_keywords),
            "replica_pid": os.getpid(),
        }


app = ModerationService.bind()
