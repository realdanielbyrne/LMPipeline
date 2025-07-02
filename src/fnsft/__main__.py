#!/usr/bin/env python3
"""
Entry point for running the SFT trainer as a module.

Usage:
    python -m fnsft.sft_trainer [arguments]
    python -m fnsft [arguments]  # This file enables this shortcut
"""

from .sft_trainer import main

if __name__ == "__main__":
    main()
