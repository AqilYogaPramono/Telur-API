from __future__ import annotations

import uuid

from googleapiclient.errors import HttpError

from APP.core.gdrive import (
    folder_id_for_class_label,
    folder_id_hasil_klasifikasi,
    http_error_detail,
    label_to_env_key,
    upload_bytes,
)


def _upload_fail_reason(error: BaseException) -> str:
    if isinstance(error, HttpError):
        return http_error_detail(error)
    return str(error)[:400]


def upload_production_analysis(
    overlay_jpeg_bytes: bytes | None,
    crop_items: list[dict[str, object]],
) -> dict[str, object]:
    batch_id = uuid.uuid4().hex
    file_ids: dict[str, str] = {}
    drive_filenames: dict[str, str] = {}
    skipped: list[str] = []

    hasil_folder = folder_id_hasil_klasifikasi()
    if overlay_jpeg_bytes and hasil_folder:
        name = f"{uuid.uuid4().hex}.jpg"
        try:
            fid = upload_bytes(hasil_folder, name, overlay_jpeg_bytes, "image/jpeg")
            file_ids["bounding_box"] = fid
            drive_filenames["bounding_box"] = name
        except Exception as error:
            skipped.append(f"bounding_box | {_upload_fail_reason(error)}")
    elif overlay_jpeg_bytes and not hasil_folder:
        skipped.append("bounding_box (HASIL_KLASIFIKASI empty)")

    for item in crop_items:
        label = str(item["label"])
        png = item["png_bytes"]
        egg_index = int(item["egg_index"])
        folder_id = folder_id_for_class_label(label)
        if not folder_id:
            key = label_to_env_key(label)
            skipped.append(f"egg_{egg_index} ({key} empty)")
            continue
        name = f"{uuid.uuid4().hex}.png"
        try:
            fid = upload_bytes(folder_id, name, png, "image/png")
            file_ids[f"egg_{egg_index}"] = fid
            drive_filenames[f"egg_{egg_index}"] = name
        except Exception as error:
            skipped.append(f"egg_{egg_index} | {_upload_fail_reason(error)}")

    if skipped:
        uploaded_any = bool(file_ids)
        if not uploaded_any:
            message = "No files were uploaded to Google Drive."
        else:
            message = "Some files failed to upload to Google Drive."
    else:
        message = "All files were uploaded to Google Drive."

    return {
        "batch_id": batch_id,
        "drive_file_ids": file_ids,
        "drive_filenames": drive_filenames,
        "drive_skipped": skipped,
        "message": message,
        "uploaded_any": bool(file_ids),
    }

