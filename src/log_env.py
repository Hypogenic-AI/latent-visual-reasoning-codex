import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy


def main() -> None:
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)


if __name__ == "__main__":
    main()
