import os
import shutil
import argparse


def delete_extra_versions(root_dir, max_version):
    for folder_name in os.listdir(root_dir):
        if folder_name.endswith("_logs"):
            log_dir = os.path.join(root_dir, folder_name)
            for version_folder in os.listdir(log_dir):
                if version_folder.startswith("version_"):
                    version = int(version_folder.split("_")[-1])
                    if version > max_version:
                        path_to_delete = os.path.join(log_dir, version_folder)
                        shutil.rmtree(path_to_delete)
                        print(f"Deleted {path_to_delete}")


def main():
    parser = argparse.ArgumentParser(description="Delete extra version folders.")
    parser.add_argument("--root-dir", type=str, help="Root directory containing log directories")
    parser.add_argument("--max-version", type=int, help="Maximum version to keep")

    args = parser.parse_args()

    delete_extra_versions(args.root_dir, args.max_version)


if __name__ == "__main__":
    main()
