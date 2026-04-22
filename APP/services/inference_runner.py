import asyncio
from collections.abc import Callable
from typing import TypeVar

from starlette.concurrency import run_in_threadpool

T = TypeVar("T")

DEFAULT_INFERENCE_TIMEOUT_SECONDS = 300


async def run_sync_with_inference_timeout(
    func: Callable[[], T],
    *,
    timeout_seconds: int = DEFAULT_INFERENCE_TIMEOUT_SECONDS,
) -> T:
    return await asyncio.wait_for(run_in_threadpool(func), timeout=timeout_seconds)
