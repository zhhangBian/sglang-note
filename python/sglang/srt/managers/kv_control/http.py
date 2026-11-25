# SPDX-License-Identifier: Apache-2.0
"""HTTP facade for the same typed scheduler control service used by Engine."""

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from sglang.srt.managers.kv_control.contracts import is_native_action
from sglang.srt.managers.kv_hints import decode_kv_hints_envelope
from sglang.srt.runtime_context import get_serving
from sglang.srt.utils.auth import AuthLevel, decide_request_auth


def authorize_attached_actions(hints, request):
    if not hints or not (admin_key := get_serving().admin_api_key):
        return
    envelopes = hints if isinstance(hints, list) else (hints,)
    try:
        native = any(
            is_native_action(action)
            for envelope in envelopes
            if envelope
            for action in decode_kv_hints_envelope(envelope).actions
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not native:
        return
    # NORMAL inference auth may already require a distinct user bearer.
    authorization = request.headers.get(
        "X-SGLang-Admin-Authorization", request.headers.get("Authorization")
    )
    decision = decide_request_auth(
        method="POST",
        path="/v1/kv/control",
        authorization_header=authorization,
        api_key=None,
        admin_api_key=admin_key,
        auth_level=AuthLevel.ADMIN_OPTIONAL,
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=403,
            detail="Native KV control actions require admin authorization",
        )


def install_routes(app, get_manager):
    async def send(command):
        reply = await get_manager().kv_control(command)
        code = reply.get("error", {}).get("code")
        status = (
            200
            if reply["success"]
            else {
                "unsupported": 503,
                "unavailable": 503,
                "capacity": 429,
                "missing": 404,
                "conflict": 409,
                "stale_epoch": 409,
                "wrong_engine": 409,
                "incompatible_namespace": 409,
                "busy": 409,
            }.get(code, 400)
        )
        return JSONResponse(reply, status_code=status)

    @app.get("/v1/kv/capabilities")
    async def capabilities():
        return await send({"kind": "capabilities"})

    @app.post("/v1/kv/control")
    async def control(command: dict):
        return await send(command)
