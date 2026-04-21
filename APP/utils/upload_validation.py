MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


class UploadValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def normalized_image_extension(filename: str) -> str:
    stripped = filename.strip()
    if not stripped:
        raise UploadValidationError("File name is required.")
    if "." not in stripped:
        raise UploadValidationError(
            "Invalid file extension. Allowed extensions are .jpg, .jpeg, and .png."
        )
    extension = f".{stripped.rsplit('.', 1)[-1].lower()}"
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise UploadValidationError(
            "Invalid file extension. Allowed extensions are .jpg, .jpeg, and .png."
        )
    return extension


def validate_image_content_type(content_type: str | None) -> None:
    if not content_type or not content_type.startswith("image/"):
        raise UploadValidationError("Only image files are allowed.")


def validate_image_byte_size(image_bytes: bytes, max_bytes: int = MAX_IMAGE_UPLOAD_BYTES) -> None:
    if not image_bytes:
        raise UploadValidationError("Uploaded file is empty.")
    if len(image_bytes) > max_bytes:
        raise UploadValidationError(
            "File is too large. Maximum allowed size is 10 MB.",
            status_code=413,
        )


def validate_image_upload(filename: str, content_type: str | None, image_bytes: bytes) -> str:
    extension = normalized_image_extension(filename)
    validate_image_content_type(content_type)
    validate_image_byte_size(image_bytes)
    return extension
