import os
import shutil
from loguru import logger
from pathlib import Path
from typing import List


def list_directories(directory: str) -> List[Path]:
    """
    Returns a list of all direct subdirectories in the given directory.
    """
    directory_path = Path(directory)
    return [subdir for subdir in directory_path.iterdir() if subdir.is_dir()]


def list_files(directory: str) -> List[Path]:
    """
    Returns a list of all file names (without the whole path) within the given directory.
    """
    directory_path = Path(directory)
    return [file for file in directory_path.iterdir() if file.is_file()]


def copy_file(source_file, destination_file):
    try:
        shutil.copy2(source_file, destination_file)
        logger.success(f"File copied successfully from {source_file} to {destination_file}")
    except FileNotFoundError:
        logger.error(f"Source file {source_file} not found.")
    except PermissionError:
        logger.error(f"Permission denied. Unable to copy the file to {destination_file}.")
    except Exception as e:
        logger.error(f"An error occurred while copying the file: {e}")


def copy_folder(source_folder, destination_folder):
    try:
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
            print(f"Existing destination folder '{destination_folder}' removed.")

        shutil.copytree(source_folder, destination_folder)
        logger.success(f"Folder copied successfully from {source_folder} to {destination_folder}")
    except FileNotFoundError:
        logger.error(f"Source folder {source_folder} not found.")
    except PermissionError:
        logger.error(f"Permission denied. Unable to copy the folder to {destination_folder}.")
    except Exception as e:
        logger.error(f"An error occurred while copying the folder: {e}")
