from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SPARK_DIST = "spark-3.5.9-bin-hadoop3"
ICEBERG_JAR_NAME = "iceberg-spark-runtime-3.5_2.12-1.11.0.jar"
CATALOG_NAME = "iceberg"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_tools_dir() -> Path:
    return project_root() / ".tools"


def bundled_iceberg_jar() -> Path:
    return project_root() / "third_party" / ICEBERG_JAR_NAME


@dataclass(frozen=True)
class SparkRuntime:
    java_home: Path
    spark_home: Path
    iceberg_jar: Path

    @property
    def spark_sql(self) -> Path:
        return self.spark_home / "bin" / "spark-sql"

    @property
    def java(self) -> Path:
        return self.java_home / "bin" / "java"


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def _jdk21_home(explicit: str | Path | None) -> Path:
    tools = project_tools_dir()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    if os.environ.get("JAVA_HOME"):
        candidates.append(Path(os.environ["JAVA_HOME"]).expanduser().resolve())
    candidates.append(tools / "jdk21")
    home = _first_existing(candidates)
    if home is None or not (home / "bin" / "java").exists():
        raise FileNotFoundError(
            "JDK 21 not found. Set --java-home / JAVA_HOME, copy or link it to "
            f"{tools / 'jdk21'}, or run `python -m iceberg_vector_loader bootstrap`."
        )
    return home


def _spark_home(explicit: str | Path | None) -> Path:
    tools = project_tools_dir()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    if os.environ.get("SPARK_HOME"):
        candidates.append(Path(os.environ["SPARK_HOME"]).expanduser().resolve())
    candidates.append(tools / SPARK_DIST)
    home = _first_existing(candidates)
    if home is None or not (home / "bin" / "spark-sql").exists():
        raise FileNotFoundError(
            "Spark 3.5.9 not found. Set --spark-home / SPARK_HOME, copy or link "
            f"{SPARK_DIST} to {tools / SPARK_DIST}, or run "
            "`python -m iceberg_vector_loader bootstrap`."
        )
    return home


def _iceberg_jar(explicit: str | Path | None) -> Path:
    tools = project_tools_dir()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    if os.environ.get("ICEBERG_SPARK_JAR"):
        candidates.append(Path(os.environ["ICEBERG_SPARK_JAR"]).expanduser().resolve())
    candidates.append(bundled_iceberg_jar())
    candidates.append(tools / ICEBERG_JAR_NAME)
    jar = _first_existing(candidates)
    if jar is None or not jar.is_file():
        raise FileNotFoundError(
            f"{ICEBERG_JAR_NAME} not found. Expected the vendored copy at "
            f"{bundled_iceberg_jar()}, or set --iceberg-jar / ICEBERG_SPARK_JAR."
        )
    return jar


def resolve_runtime(
    java_home: str | Path | None = None,
    spark_home: str | Path | None = None,
    iceberg_jar: str | Path | None = None,
) -> SparkRuntime:
    runtime = SparkRuntime(
        java_home=_jdk21_home(java_home),
        spark_home=_spark_home(spark_home),
        iceberg_jar=_iceberg_jar(iceberg_jar),
    )
    return runtime


def runtime_available() -> bool:
    try:
        resolve_runtime()
        return True
    except FileNotFoundError:
        return False
