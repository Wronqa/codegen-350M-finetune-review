from __future__ import annotations

import os
import sys
import json
import time
import pathlib
import mimetypes
import logging
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Iterable, Tuple, Optional
from urllib.parse import quote_plus

import boto3
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

load_dotenv()

log = logging.getLogger("s3uploader")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")
    bucket, _, prefix = s3_uri[5:].partition("/")
    if not bucket:
        raise ValueError("Invalid S3 URI (missing bucket)")
    return bucket, prefix.rstrip("/")

def _tag_header(tags: Optional[Dict[str, str]]) -> Optional[str]:
    if not tags:
        return None
    parts = []
    for k, v in tags.items():
        if k is None or v is None:
            continue
        parts.append(f"{quote_plus(str(k))}={quote_plus(str(v))}")
    return "&".join(parts) if parts else None


def _should_exclude(path: pathlib.Path, patterns: Iterable[str]) -> bool:
    p = path.as_posix()
    for pat in patterns:
        pat = pat.strip()
        if not pat:
            continue
        if path.match(pat) or p.endswith(pat):
            return True
    return False

def upload_dir_to_s3(
    local_dir: str,
    s3_uri: str,
    extra_tags: dict | None = None,
) -> str:
    bucket, base_prefix = _parse_s3_uri(s3_uri)

    p = pathlib.Path(local_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Local adapter dir not found: {local_dir}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    version_prefix = f"{base_prefix}/{ts}" if base_prefix else ts

    tcfg = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=8 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )

    s3 = boto3.client("s3")

    extra_args: Dict[str, str] = {}
    acl = os.getenv("ADAPTER_S3_ACL")
    if acl:
        extra_args["ACL"] = acl

    sse = os.getenv("ADAPTER_S3_SSE")
    if sse:
        extra_args["ServerSideEncryption"] = sse
        kms_key = os.getenv("ADAPTER_S3_KMS_KEY_ID")
        if sse == "aws:kms" and kms_key:
            extra_args["SSEKMSKeyId"] = kms_key

    cache_control = os.getenv("ADAPTER_S3_CACHE_CONTROL")
    if cache_control:
        extra_args["CacheControl"] = cache_control

    tag_header = _tag_header(extra_tags)
    if tag_header:
        extra_args["Tagging"] = tag_header

    exclude = [x for x in (os.getenv("ADAPTER_S3_EXCLUDE", "").split(",")) if x.strip()]
    uploaded = 0
    errors = 0
    files_list: list[str] = []

    for path in p.rglob("*"):
        if path.is_dir():
            continue
        if _should_exclude(path, exclude):
            log.debug("exclude: %s", path)
            continue

        rel = path.relative_to(p).as_posix()
        key = f"{version_prefix}/{rel}"

        ctype, _ = mimetypes.guess_type(path.as_posix())
        ea = dict(extra_args)  
        if ctype:
            ea["ContentType"] = ctype

        try:
            s3.upload_file(
                Filename=str(path),
                Bucket=bucket,
                Key=key,
                ExtraArgs=ea,
                Config=tcfg,
            )
            uploaded += 1
            files_list.append(rel)
        except ClientError as e:
            errors += 1
            log.error("upload failed: %s → s3://%s/%s (%s)", path, bucket, key, e)

    manifest = {
        "bucket": bucket,
        "prefix": version_prefix,
        "files_count": uploaded,
        "errors": errors,
        "created_at": ts,
        "model_id": os.getenv("MODEL_ID", ""),
    }
    try:
        man_args = dict(extra_args)
        man_args["ContentType"] = "application/json"
        s3.put_object(
            Bucket=bucket,
            Key=f"{version_prefix}/manifest.json",
            Body=json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
            **({"Tagging": tag_header} if tag_header else {}),
            **({"ServerSideEncryption": sse} if sse else {}),
            **({"SSEKMSKeyId": os.getenv("ADAPTER_S3_KMS_KEY_ID")} if sse == "aws:kms" and os.getenv("ADAPTER_S3_KMS_KEY_ID") else {}),
            **({"CacheControl": cache_control} if cache_control else {}),
            **({"ACL": acl} if acl else {}),
            ContentType="application/json",
        )
    except ClientError as e:
        errors += 1
        log.error("manifest upload failed: s3://%s/%s (%s)", bucket, f"{version_prefix}/manifest.json", e)

    if errors:
        log.warning("completed with %d error(s) — uploaded=%d", errors, uploaded)
    else:
        log.info("uploaded=%d file(s) → s3://%s/%s", uploaded, bucket, version_prefix)

    return f"s3://{bucket}/{version_prefix}"

if __name__ == "__main__":
    local_dir = sys.argv[1] if len(sys.argv) > 1 else "adapters/codegen350m_lora"
    s3_uri = os.environ.get("ADAPTER_S3_URI", "")
    if not s3_uri:
        print("ADAPTER_S3_URI env is empty; skip upload.", file=sys.stderr)
        sys.exit(0)
    dest = upload_dir_to_s3(local_dir, s3_uri)
    print(f"Uploaded adapter to: {dest}")
