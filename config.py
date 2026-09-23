"""Defaults for the original MDFNet training implementation."""
import os

process_data_dir = os.environ.get("MDFNET_DATA_DIR", "./data")
lr = 1e-3
weight_decay = 0.01
