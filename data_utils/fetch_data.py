"""Fetch the nuPlan bin sets from S3.

Ported from upstream 3.0. Datasets are declared in data_utils/datasets.yaml
and sync into <repo>/data/<name>/ (gitignored). Fetching elsewhere
(--data-root) means pointing env.map_dir there too.

    python data_utils/fetch_data.py                  # nuplan_mini_val, ~10 GB
    python data_utils/fetch_data.py --list
    python data_utils/fetch_data.py nuplan_val
    python data_utils/fetch_data.py nuplan_mini_val --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(REPO_ROOT, "data_utils", "datasets.yaml")
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data")
REQUIRED_FIELDS = ("s3_uri", "description", "license", "size")


def load_manifest() -> dict[str, dict]:
    """
    Read datasets.yaml and check that every entry has the required fields.

    Exits with an error message on an empty manifest or a missing field.

    Returns:
        dict[str, dict]: dataset name -> manifest entry.
    """
    with open(MANIFEST_PATH) as manifest_file:
        manifest = yaml.safe_load(manifest_file)
    if not isinstance(manifest, dict) or not manifest:
        sys.exit(
            f"error: {MANIFEST_PATH} is empty or not a mapping of dataset"
            " entries"
        )
    for name, entry in manifest.items():
        for field in REQUIRED_FIELDS:
            if field not in entry:
                sys.exit(
                    f"error: dataset '{name}' in {MANIFEST_PATH} is missing"
                    f" the '{field}' field"
                )
    return manifest


def list_datasets(
        manifest: dict[str, dict],
        data_root: str,
    ) -> None:
    """
    Print every dataset with its source, size, license and local status.

    Args:
        manifest (dict[str, dict]): dataset name -> manifest entry.
        data_root (str): directory holding one folder per dataset.
    """
    for name, entry in manifest.items():
        local_dir = os.path.join(data_root, name)
        # aws s3 sync creates the destination dir even when it fails, so an
        # empty dir does not count as fetched.
        is_fetched = os.path.isdir(local_dir) and os.listdir(local_dir)
        status = "present" if is_fetched else "not fetched"
        default_marker = "  (default)" if entry.get("default") else ""
        print(f"{name}  [{status}]{default_marker}")
        print(f"    source:  {entry['s3_uri']}")
        print(f"    size:    {entry['size']}")
        print(f"    local:   {local_dir}")
        print(f"    about:   {entry['description']}")
        print(f"    license: {entry['license']}")


def fetch_dataset(
        manifest: dict[str, dict],
        name: str,
        data_root: str,
        dry_run: bool,
    ) -> None:
    """
    Sync one dataset from S3 into <data_root>/<name>/ with the AWS CLI.

    Exits with the aws exit code on failure. A sync is incremental, so a
    rerun only fetches missing or changed files.

    Args:
        manifest (dict[str, dict]): dataset name -> manifest entry.
        name (str): dataset to fetch.
        data_root (str): directory holding one folder per dataset.
        dry_run (bool): only list what would be downloaded.
    """
    if name not in manifest:
        sys.exit(
            f"error: unknown dataset '{name}'. Available:"
            f" {', '.join(sorted(manifest))}"
        )
    if shutil.which("aws") is None:
        sys.exit(
            "error: the AWS CLI is not on PATH. Install it in the venv:"
            " uv pip install awscli (see docs/setup.md)"
        )
    entry = manifest[name]
    destination_dir = os.path.join(data_root, name)
    sync_command = ["aws", "s3", "sync", entry["s3_uri"], destination_dir]
    # The AWS CLI refuses even public buckets without credentials unless the
    # request is explicitly unsigned.
    if entry.get("public"):
        sync_command.append("--no-sign-request")
    if dry_run:
        sync_command.append("--dryrun")
    print(f"license: {entry['license']}")
    print(f"size:    {entry['size']}")
    print(f"syncing {entry['s3_uri']} -> {destination_dir}")
    completed = subprocess.run(sync_command)
    if completed.returncode != 0:
        sys.exit(f"error: aws s3 sync exited with {completed.returncode}.")
    if not dry_run:
        file_count = sum(
            len(files) for _, _, files in os.walk(destination_dir)
        )
        print(f"done: {file_count} files in {destination_dir}")


def main() -> None:
    """Parse the command line, then list or fetch the named datasets."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=(
            "dataset names from data_utils/datasets.yaml; none = the entries"
            " marked default"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list datasets and their local status",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="destination root (default: <repo>/data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be downloaded without writing",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    if args.list:
        list_datasets(manifest, args.data_root)
        return
    dataset_names = args.datasets
    if not dataset_names:
        dataset_names = [
            name for name, entry in manifest.items() if entry.get("default")
        ]
        if not dataset_names:
            sys.exit(
                f"error: no dataset given and no entry in {MANIFEST_PATH} is"
                " marked default: true"
            )
        defaults_text = ", ".join(dataset_names)
        print(f"no dataset given, fetching the defaults: {defaults_text}")
    for name in dataset_names:
        fetch_dataset(manifest, name, args.data_root, args.dry_run)


if __name__ == "__main__":
    main()
