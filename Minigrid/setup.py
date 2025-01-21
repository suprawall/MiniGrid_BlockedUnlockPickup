"""Setups the project."""

from __future__ import annotations

import pathlib

from setuptools import setup, find_packages

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the minigrid version."""
    path = CWD / "minigrid" / "__init__.py"
    content = path.read_text(encoding="utf-8")  # Ajout de l'encodage explicite

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


def get_description():
    """Gets the description from the readme."""
    with open("README.md", encoding="utf-8") as fh:  # Ajout de l'encodage explicite
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return long_description


setup(name="minigrid", version=get_version(), packages=find_packages(), long_description=get_description())
