## upload_artifacts.py
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client


def get_supabase_client() -> Client:
    load_dotenv()

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]  # service role key
    return create_client(url, key)


def upload_file(sb: Client, bucket: str, local_path: Path, dest_path: str):
    """
    Upload a single file to Supabase Storage.

    local_path: local filesystem path
    dest_path:  path inside the bucket (e.g. "rf_ks_predictions.csv")
    """
    print(f"Uploading {local_path} -> {bucket}/{dest_path}")

    with open(local_path, "rb") as f:
        # upsert=True to overwrite if it already exists
        sb.storage.from_(bucket).upload(
            dest_path,
            f,
            {"cacheControl": "3600", "upsert": "true"}
        )


def main():
    sb = get_supabase_client()
    bucket = os.environ.get("SUPABASE_BUCKET", "artifacts")

    project_root = Path(__file__).resolve().parent

    # 1) All files inside local "artifacts/" folder
    artifacts_dir = project_root / "artifacts"
    if artifacts_dir.exists():
        for p in artifacts_dir.rglob("*"):
            if p.is_file():
                # Put them in the bucket root using just the filename
                dest = p.name
                upload_file(sb, bucket, p, dest)
    else:
        print(f"WARNING: {artifacts_dir} does not exist")

    # 2) Specific extra files (e.g. data/3_df_merged_cleaned.csv)
    extra_files = [
        project_root / "data" / "3_df_merged_cleaned.csv",
    ]

    for p in extra_files:
        if p.exists():
            dest = p.name  # store in bucket root as "3_df_merged_cleaned.csv"
            upload_file(sb, bucket, p, dest)
        else:
            print(f"WARNING: extra file not found: {p}")


if __name__ == "__main__":
    main()
