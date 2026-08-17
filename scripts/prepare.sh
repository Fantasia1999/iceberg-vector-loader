#!/usr/bin/env bash
# Resolve JDK 21 and Spark 3.5.9-bin-hadoop3.
# Prefer an existing JAVA_HOME / SPARK_HOME, then a copy or symlink under .tools/,
# and only then download into .tools/.
# The Iceberg Spark runtime jar is vendored at third_party/ and is not downloaded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="${ICEBERG_VECTOR_TOOLS_DIR:-$ROOT/.tools}"
SPARK_DIST="spark-3.5.9-bin-hadoop3"
SPARK_TGZ="$TOOLS/${SPARK_DIST}.tgz"
SPARK_HOME_DIR="$TOOLS/$SPARK_DIST"
SPARK_URL="https://archive.apache.org/dist/spark/spark-3.5.9/spark-3.5.9-bin-hadoop3.tgz"
JDK_HOME_DIR="$TOOLS/jdk21"
JDK_TGZ="$TOOLS/jdk21.tar.gz"
JDK_URL="https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse?project=jdk"
ICEBERG_JAR="$ROOT/third_party/iceberg-spark-runtime-3.5_2.12-1.11.0.jar"

download() {
  local url="$1"
  local dest="$2"
  if [[ -f "$dest" && -s "$dest" ]]; then
    echo "keep $dest"
    return
  fi
  echo "download $url"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 3 --retry-delay 2 -o "${dest}.partial" "$url"
  mv "${dest}.partial" "$dest"
}

if [[ ! -f "$ICEBERG_JAR" ]]; then
  echo "missing vendored Iceberg jar: $ICEBERG_JAR" >&2
  exit 1
fi

mkdir -p "$TOOLS"

if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
  echo "use JAVA_HOME=$JAVA_HOME"
elif [[ -x "$JDK_HOME_DIR/bin/java" ]]; then
  echo "keep $JDK_HOME_DIR"
else
  download "$JDK_URL" "$JDK_TGZ"
  echo "extract $JDK_TGZ"
  rm -rf "$TOOLS/_jdk_extract"
  mkdir -p "$TOOLS/_jdk_extract"
  tar -xzf "$JDK_TGZ" -C "$TOOLS/_jdk_extract"
  extracted="$(find "$TOOLS/_jdk_extract" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  rm -rf "$JDK_HOME_DIR"
  mv "$extracted" "$JDK_HOME_DIR"
  rm -rf "$TOOLS/_jdk_extract"
fi

if [[ -n "${SPARK_HOME:-}" && -x "${SPARK_HOME}/bin/spark-sql" ]]; then
  echo "use SPARK_HOME=$SPARK_HOME"
elif [[ -x "$SPARK_HOME_DIR/bin/spark-sql" ]]; then
  echo "keep $SPARK_HOME_DIR"
else
  download "$SPARK_URL" "$SPARK_TGZ"
  echo "extract $SPARK_TGZ"
  tar -xzf "$SPARK_TGZ" -C "$TOOLS"
fi

resolved_java="${JAVA_HOME:-$JDK_HOME_DIR}"
resolved_spark="${SPARK_HOME:-$SPARK_HOME_DIR}"
if [[ ! -x "$resolved_java/bin/java" ]]; then
  echo "JDK 21 not found. Set JAVA_HOME, copy/link it to $JDK_HOME_DIR, or let this script download it." >&2
  exit 1
fi
if [[ ! -x "$resolved_spark/bin/spark-sql" ]]; then
  echo "Spark 3.5.9 not found. Set SPARK_HOME, copy/link it to $SPARK_HOME_DIR, or let this script download it." >&2
  exit 1
fi

echo
"$resolved_java/bin/java" -version
echo "JAVA_HOME=$resolved_java"
echo "SPARK_HOME=$resolved_spark"
echo "ICEBERG_SPARK_JAR=$ICEBERG_JAR"
echo "prepare ok"
