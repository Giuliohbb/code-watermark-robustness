"""Create the tiny offline model used by configs/smoke.yaml.

Usage: python scripts/make_tiny_model.py [output_dir]
"""

import sys

from cwm.testing import build_tiny_model

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else ".cache/tiny-model"
    print(f"tiny model ready at {build_tiny_model(target)}")
