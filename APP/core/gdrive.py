from __future__ import annotations

import io
import os
from dataclasses import dataclass

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload

DRIVE_SCOPES = ("https://www.googleapis.com/auth/drive",)
TOKEN_URI = "https://oauth2.googleapis.com/token"

_drive = None
_creds: Credentials | None = None


def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set.")
    return value


def _load_oauth_credentials() -> Credentials:
    creds = Credentials(
        token=None,
        refresh_token=_required_env("GDRIVE_OAUTH_REFRESH_TOKEN"),
        token_uri=TOKEN_URI,
        client_id=_required_env("GDRIVE_OAUTH_CLIENT_ID"),
        client_secret=_required_env("GDRIVE_OAUTH_CLIENT_SECRET"),
        scopes=list(DRIVE_SCOPES),
    )
    creds.refresh(Request())
    return creds


def get_drive():
    global _drive, _creds
    if _creds is None:
        _creds = _load_oauth_credentials()
    if not _creds.valid or _creds.expired:
        _creds.refresh(Request())
    if _drive is None:
        _drive = build("drive", "v3", credentials=_creds, cache_discovery=False)
    return _drive


def drive_file_view_url(file_id: str) -> str:
    return f"https://drive.google.com/file/d/{file_id}/view"


def folder_id_hasil_klasifikasi() -> str | None:
    value = (os.getenv("HASIL_KLASIFIKASI") or "").strip()
    return value or None


def label_to_env_key(label: str) -> str:
    return label.strip().upper().replace(" ", "_").replace("-", "_")


def folder_id_for_class_label(label: str) -> str | None:
    key = label_to_env_key(label)
    value = (os.getenv(key) or "").strip()
    return value or None


def upload_bytes(folder_id: str, filename: str, data: bytes, mime_type: str) -> str:
    drive = get_drive()
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=False)
    body = {"name": filename, "parents": [folder_id]}
    created = (
        drive.files()
        .create(
            body=body,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        )
        .execute()
    )
    return created["id"]


def http_error_detail(error: HttpError) -> str:
    try:
        content = error.content.decode("utf-8", errors="replace") if error.content else ""
        return f"{error.resp.status} {error.resp.reason}: {content[:500]}"
    except Exception:
        return str(error)


@dataclass(frozen=True)
class GDriveUploadResult:
    file_ids: dict[str, str]
    skipped: tuple[str, ...]

