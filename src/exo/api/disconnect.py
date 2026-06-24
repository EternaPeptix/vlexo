import asyncio

from anyio import BrokenResourceError, ClosedResourceError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_CLIENT_DISCONNECT_ERRORS = (
    BrokenResourceError,
    ClosedResourceError,
    ConnectionError,
    ConnectionResetError,
    TimeoutError,
    asyncio.TimeoutError,
)


class DisconnectTolerantMiddleware:
    """Ignore send failures after the client disconnects mid-response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        disconnected = False

        async def send_wrapper(message: Message) -> None:
            nonlocal disconnected
            if disconnected:
                return
            try:
                await send(message)
            except _CLIENT_DISCONNECT_ERRORS:
                disconnected = True

        await self.app(scope, receive, send_wrapper)
