import asyncio

from starlette.concurrency import run_in_threadpool

from APP.services.egg_classification_experiment import (
    EggCNNClassification,
    classify_egg_from_image_bytes,
)

INFERENCE_TIMEOUT_SECONDS = 300


async def run_direct_cnn_classification(image_bytes: bytes) -> EggCNNClassification:
    return await asyncio.wait_for(
        run_in_threadpool(classify_egg_from_image_bytes, image_bytes),
        timeout=INFERENCE_TIMEOUT_SECONDS,
    )
