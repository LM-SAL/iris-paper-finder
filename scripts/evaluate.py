"""Compatibility delegate for the unified ``iris-papers evaluate`` command."""

from __future__ import annotations

import sys

from paper_data_linking.cli import main


if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
