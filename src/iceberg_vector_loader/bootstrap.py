from __future__ import annotations

import subprocess
from pathlib import Path

from iceberg_vector_loader.spark_env import project_root, project_tools_dir


def bootstrap_tools(tools_dir: Path | None = None) -> Path:
    """Run scripts/prepare.sh to fetch JDK 21 and Spark 3.5.9."""
    script = project_root() / "scripts" / "prepare.sh"
    if not script.is_file():
        raise FileNotFoundError(f"missing prepare script: {script}")
    env = None
    if tools_dir is not None:
        import os

        env = os.environ.copy()
        env["ICEBERG_VECTOR_TOOLS_DIR"] = str(tools_dir)
    subprocess.run(["bash", str(script)], check=True, env=env)
    return Path(tools_dir) if tools_dir is not None else project_tools_dir()
