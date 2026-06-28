from collections.abc import AsyncIterator
from typing import Final

import anyio

_DONE: Final = object()

# OpenAI-compatible SSE frame with an ``exo`` vendor extension.  Hermes and
# other first-party clients use this to distinguish transport liveness from
# real model output during long prefill.
SEMANTIC_SSE_KEEPALIVE = (
    'data: {"id":"exo-keepalive","object":"chat.completion.chunk","created":0,'
    '"model":"exo","choices":[{"index":0,"delta":{"role":"assistant"},'
    '"finish_reason":null}],"exo":{"status":"waiting"}}\n\n'
)


async def with_sse_keepalive(
    generator: AsyncIterator[str],
    keepalive_message: str = ": keep-alive\n\n",
    semantic_keepalive_message: str | None = None,
    interval: float = 5.0,
) -> AsyncIterator[str]:
    ping = semantic_keepalive_message or keepalive_message
    yield ping
    send, recv = anyio.create_memory_object_stream[str | object]()

    async def _consume() -> None:
        async for item in generator:
            await send.send(item)
        await send.send(_DONE)

    async with anyio.create_task_group() as tg:
        tg.start_soon(_consume)
        while True:
            item: str | object | None = None
            with anyio.move_on_after(interval):
                item = await recv.receive()
            if item is None:
                yield ping
            elif item is _DONE:
                break
            else:
                assert isinstance(item, str)
                yield item
