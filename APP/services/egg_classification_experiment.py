import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH = _REPO_ROOT / "models" / "model_cnn_v2.h5"
_CLASS_NAMES_PATH = _REPO_ROOT / "models" / "class_names.json"


@dataclass(frozen=True)
class EggCNNClassification:
    classification_label: str
    confidence_score: float
    is_fertile: bool
    is_mati: bool


def _read_class_names() -> list[str]:
    if not _CLASS_NAMES_PATH.is_file():
        raise ValueError("Class names file was not found at models/class_names.json.")
    with open(_CLASS_NAMES_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError("Class names file is invalid. It must be a non-empty JSON array.")
    return [str(item) for item in data]


def _load_cnn_and_classes():
    if not _MODEL_PATH.is_file():
        raise ValueError("CNN model file was not found at models/model_cnn_v2.h5.")
    from tensorflow.keras.models import load_model

    model = load_model(str(_MODEL_PATH))
    names = _read_class_names()
    return model, names


_cnn_model, _class_names = _load_cnn_and_classes()
_cnn_model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)


def classify_egg_from_image_bytes(image_bytes: bytes) -> EggCNNClassification:
    import tensorflow as tf

    model = _cnn_model
    names = _class_names
    image_tensor = tf.image.decode_image(tf.constant(image_bytes), channels=3)
    image_tensor = tf.image.resize(image_tensor, [224, 224])
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    predictions = model.predict(image_tensor, verbose=0)
    class_index = int(np.argmax(predictions))
    label = names[class_index]
    confidence = float(predictions[0][class_index])
    is_fertile = label.startswith("Fertil_")
    is_mati = label.startswith("Mati_")
    return EggCNNClassification(
        classification_label=label,
        confidence_score=confidence,
        is_fertile=is_fertile,
        is_mati=is_mati,
    )
