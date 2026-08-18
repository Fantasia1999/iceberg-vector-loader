# iceberg-vector-loader

把本地向量数据导入 **Iceberg v3** 表。写入走 **Spark 原生**（`spark-sql` + Hadoop catalog），不是 pyiceberg。

适用场景：SIFT1M / SIFT10M 以及其他本地 embedding 数据集。namespace、表名都可以指定。

| 组件 | 版本 |
|---|---|
| Spark | 3.5.9-bin-hadoop3 |
| Iceberg | `iceberg-spark-runtime-3.5_2.12-1.11.0.jar`（已随仓库提交） |
| JDK | 21 |
| Python | ≥ 3.11 且 < 3.15（3.11 / 3.12 / 3.13 / 3.14） |

入口：`python -m iceberg_vector_loader <命令>`，或 `iceberg-vector-loader <命令>`。

| 命令 | 作用 | 是否需要 JDK / Spark |
|---|---|---|
| `convert` | 只把 `.fvecs` 转成 parquet | 否 |
| `load` | 准备输入并写入 Iceberg v3 表 | 是 |
| `bootstrap` | 解析 / 下载 JDK 21 和 Spark 3.5.9 | 下载时需要网络 |

## 架构

```
.fvecs                          .parquet / parquet 目录
   │                                    │
   ▼                                    │
convert（可选，只出 parquet）            │
   │                                    │
   └──────────────┬─────────────────────┘
                  ▼
            load：Python 准备输入
            （fvecs 先转 parquet；parquet 必须已有 id）
                  │
                  ▼
            spark-sql + Iceberg 1.11 Java runtime
            CREATE NAMESPACE / CREATE TABLE (format-version=3) / INSERT
                  │
                  ▼
            Hadoop catalog
            <warehouse>/<namespace>/<table>/
```

Python 只做数据准备。建表由 Spark 的 Iceberg Java writer 完成，因此是满血 V3（含 `next-row-id` 等）。表已存在且未指定 `--overwrite` 时直接退出，不会 append。

## 安装

### Python

所有命令都需要：

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### JDK 21 和 Spark 3.5.9

只 `convert` 的话到这里就可以了。`load` 还需要 **JDK 21** 和 **Spark 3.5.9-bin-hadoop3**。本机已经有的话不必下载，任选下面一种即可。Iceberg runtime jar 已在仓库里（`third_party/iceberg-spark-runtime-3.5_2.12-1.11.0.jar`），一般不用另配。

查找顺序（从高到低）：

1. CLI：`--java-home`、`--spark-home`、`--iceberg-jar`
2. 环境变量
3. `.tools/` 下的约定目录

#### 方式 A：设环境变量（推荐本机已有安装时）

要求：`$JAVA_HOME/bin/java`、`$SPARK_HOME/bin/spark-sql` 可执行。不必把它们加进 `PATH`。

```bash
export JAVA_HOME=/path/to/jdk21
export SPARK_HOME=/path/to/spark-3.5.9-bin-hadoop3
```

| 变量 | 可选 | 作用 |
|---|---|---|
| `JAVA_HOME` | 本机已有 JDK 21 时设 | JDK 根目录 |
| `SPARK_HOME` | 本机已有 Spark 3.5.9 时设 | Spark 根目录 |
| `ICEBERG_SPARK_JAR` | 一般不用 | 覆盖仓库内 Iceberg jar |
| `ICEBERG_VECTOR_TOOLS_DIR` | 一般不用 | `prepare.sh` 的下载目录，默认 `<仓库>/.tools` |

单次覆盖、不改环境也可以，见 [`load`](#load)。

#### 方式 B：拷到或链到 `.tools/`

目录名必须和约定一致（`.tools/` 已 gitignore）：

```bash
mkdir -p .tools
ln -s /path/to/jdk21 .tools/jdk21
ln -s /path/to/spark-3.5.9-bin-hadoop3 .tools/spark-3.5.9-bin-hadoop3
# 也可以 cp -a，效果相同
```

- `.tools/jdk21/bin/java`
- `.tools/spark-3.5.9-bin-hadoop3/bin/spark-sql`

#### 方式 C：本机没有再下载

```bash
./scripts/prepare.sh
# 等价：python -m iceberg_vector_loader bootstrap
```

会把 Temurin JDK 21 和 Spark 3.5.9-bin-hadoop3 下到 `.tools/`（已存在则跳过）。若已设置可用的 `JAVA_HOME` / `SPARK_HOME`，对应那一项也不会再下。

## convert

只把 TexMex `.fvecs` 转成 parquet，**不建表、不启动 Spark**。写出两列：`id`（`int64`，默认 `0 .. n-1`）和 `embedding`（`list<float32>`）。

```bash
python -m iceberg_vector_loader convert \
  --input /path/to/sift_base.fvecs \
  --output /path/to/sift_base.parquet
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | 必填 | TexMex `.fvecs`：每条 `int32 dim + dim * float32` |
| `--output` | 必填 | 目标 parquet 路径 |
| `--id-offset` | `0` | 起始 id，写成 `id_offset .. id_offset+n-1` |
| `--batch-size` | `250000` | 每批行数，用来限制转换时的内存 |

`load` 遇到 `.fvecs` 时走同一套转换。已经转好的 parquet 可以直接给 `load --input`。

## load

准备输入并写入 Iceberg v3 表。`--input` 可以是：

| 类型 | 说明 |
|---|---|
| `.fvecs` | 先转成 parquet，再导入 |
| `.parquet` / `.parq` | 直接读 footer 推断列 |
| 目录 | 当作一组 parquet 文件（Spark `parquet.\`dir\``） |

成功后打印表标识、`format-version`、location、维度、写入行数等。

### 例子

SIFT1M（fvecs，内部会先转 parquet）：

```bash
python -m iceberg_vector_loader load \
  --namespace bench \
  --table sift1m \
  --input /path/to/sift1m/sift_base.fvecs \
  --warehouse ./warehouse
```

已有 parquet（列名已是 `id` + `embedding`）：

```bash
python -m iceberg_vector_loader load \
  --namespace bench \
  --table sift10m \
  --input /data/sift10m.parquet \
  --warehouse ./warehouse \
  --driver-memory 16g
```

表已存在时默认拒绝；要重建：

```bash
python -m iceberg_vector_loader load \
  --namespace bench --table sift1m \
  --input /path/to/sift_base.fvecs \
  --warehouse ./warehouse \
  --overwrite
```

指定本机 JDK / Spark，不改环境变量：

```bash
python -m iceberg_vector_loader load \
  --java-home /path/to/jdk21 \
  --spark-home /path/to/spark-3.5.9-bin-hadoop3 \
  --namespace bench --table sift1m \
  --input /path/to/sift_base.fvecs
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--namespace` | 必填 | Iceberg namespace |
| `--table` | 必填 | 表名，不能含 `.` |
| `--input` | 必填 | `.fvecs` / `.parquet` / parquet 目录 |
| `--warehouse` | `warehouse` | 本地 Hadoop catalog 根目录 |
| `--output` | 见下文 | fvecs 转换或规范化后的 parquet 落盘路径 |
| `--overwrite` | 关 | 先 `DROP TABLE ... PURGE` 再建表。不加此参数时，表已存在则直接报错退出 |
| `--path-style` | `path` | `path`：metadata 里写绝对路径，不带 `file://`；`uri`：写成 `file:///...` |
| `--id-column` | 自动推断 | parquet 里的整数 id 列 |
| `--embedding-column` | 自动推断 | parquet 里的 `list<float>` 向量列 |
| `--id-offset` | `0` | 仅用于 fvecs 转 parquet 时的起始 id；parquet 缺 id 会直接报错 |
| `--batch-size` | `250000` | fvecs 转换 / 规范化时每批行数 |
| `--java-home` | `JAVA_HOME` 或 `.tools/jdk21` | JDK 21 |
| `--spark-home` | `SPARK_HOME` 或 `.tools/spark-3.5.9-bin-hadoop3` | Spark 3.5.9 |
| `--iceberg-jar` | `ICEBERG_SPARK_JAR` 或 `third_party/...jar` | Iceberg runtime |
| `--driver-memory` | `4g` | `spark-sql` driver 堆内存。当前是 `local[*]`，导入主要吃这块 |
| `--compression-codec` | `zstd` | Parquet 压缩：`zstd` / `snappy` / `gzip` / `lz4` / `brotli` / `uncompressed` |

`--output` 默认：

- `.fvecs`：源文件同目录的 `<stem>.parquet`
- parquet 需要改列名时：warehouse 下 `_staging/<namespace>/<table>/prepared.parquet`

### `--driver-memory` 怎么估

原始向量大约是 `行数 × 维度 × 4` 字节（float32）。Spark 里 `ARRAY<FLOAT>` 不是紧凑矩阵，再加上读 parquet、Iceberg 写缓冲，**driver 建议按原始体积的约 3 倍**，且不要低于 `4g`。

`推荐 ≈ max(4g, 3 × 行数 × 维度 × 4 / 1024³)g`，再向上取整到 2 的幂（4 / 8 / 16 / 32 / 64）。

128 维（SIFT / 同类 fvecs）：

| 规模 | 原始体积 | 推荐 `--driver-memory` |
|---|---|---|
| 10 万 | ~50 MB | `4g`（默认） |
| **100 万（SIFT1M）** | ~0.5 GB | **`4g`（默认，已实测）** |
| **1000 万（SIFT10M）** | ~5 GB | **`16g`** |
| 1 亿 | ~51 GB | `64g`，或不要用单机 `local[*]` |

768 维（常见文本 embedding）：

| 规模 | 原始体积 | 推荐 `--driver-memory` |
|---|---|---|
| 10 万 | ~0.3 GB | `4g` |
| 100 万 | ~3 GB | `8g`～`16g` |
| 1000 万 | ~29 GB | `32g`～`64g` |

1024 维大约再乘 1024/768 ≈ 1.3。机器内存要大于该值，并给操作系统和 Python 转 fvecs 留出余量（SIFT10M 转换阶段还会再占一份 numpy 缓冲，由 `--batch-size` 限制）。

OOM 时先加大 `--driver-memory`，不要先改 `--batch-size`（后者只影响 Python 准备输入，几乎不影响 Spark driver）。

### 表结构

固定两列，其余 parquet 列原样带上：

| 列 | Spark / Iceberg 类型 | 说明 |
|---|---|---|
| `id` | `BIGINT NOT NULL` | 主键；fvecs 默认 `0 .. n-1` |
| `embedding` | `ARRAY<FLOAT> NOT NULL` | 对应 Spark `array<float>` |
| 其他列 | 按 Arrow 类型映射 | 如 `label STRING` |

表属性：

- `format-version=3`
- `write.format.default=parquet`
- `write.parquet.compression-codec`（默认 `zstd`，可用 `--compression-codec` 改）
- `iceberg.vector.dimension=<推断出的维度>`

Catalog 是 Hadoop 风格：

```text
<warehouse>/<namespace>/<table>/
  data/*.parquet
  metadata/vN.metadata.json
  metadata/version-hint.text
```

Spark 读同一张表可以这样配：

```
spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions
spark.sql.catalog.iceberg=org.apache.iceberg.spark.SparkCatalog
spark.sql.catalog.iceberg.type=hadoop
spark.sql.catalog.iceberg.warehouse=/绝对路径/warehouse
```

然后：

```sql
SELECT id, embedding FROM iceberg.bench.sift1m LIMIT 5;
```

### 列名怎么推断

Parquet footer 里能读到 schema，一般不用手写列名。

**embedding 列**

1. `--embedding-column` 指定则用指定列（必须是 `list` / `large_list` / `fixed_size_list` of float）
2. 否则在 float 数组列里找名为 `embedding` / `vector` / `features` / `emb` 的
3. 只有一列 float 数组就用它
4. 多列且名字对不上：报错，要求 `--embedding-column`

**id 列**

1. `--id-column` 指定则用指定整数列
2. 否则找名为 `id` / `index` / `idx` 的整数列
3. 只有一列整数就用它
4. 没有整数列：**直接报错退出**，不会自动生成 id
5. 多列整数且名字对不上：报错，要求 `--id-column`

`.fvecs` 本身没有 id 字段，转换阶段仍会写入 `0 .. n-1`（可用 `--id-offset` 调整）。这和「parquet 缺 id」不是同一条路径。

embedding 会规范成 `list<float32>`。维度必须整列一致，否则报错。

### 路径风格

Iceberg metadata 里的 `location`：

- 默认 `--path-style path`：`/data/warehouse/bench/sift1m`（不带 `file://`）
- `--path-style uri`：`file:///data/warehouse/bench/sift1m`

相对路径会先按当前工作目录展开成绝对路径再写入。

## 测试

```bash
# load 集成测试需要 JDK + Spark：本机已有则 export，或链到 .tools/；没有再跑 prepare.sh
uv pip install -e ".[dev]"
pytest
```

没有可用的 JDK/Spark（环境变量或 `.tools/`）时，不依赖 Spark 的单测仍会跑；需要起 `spark-sql` 的用例会被跳过。

## 故障排查

- **找不到 JDK / Spark / jar**：只影响 `load`。先确认已 `export JAVA_HOME` / `SPARK_HOME`，或 `.tools/jdk21/bin/java`、`.tools/spark-3.5.9-bin-hadoop3/bin/spark-sql` 存在。本机没有再跑 `./scripts/prepare.sh`。Iceberg jar 应在 `third_party/iceberg-spark-runtime-3.5_2.12-1.11.0.jar`。
- **spark-sql 失败**：看 warehouse 下 `_staging/<namespace>/<table>/spark/spark-sql.log`。
- **OOM**：加大 `--driver-memory`。SIFT1M 用默认 `4g` 即可，SIFT10M 建议 `16g`。
- **多列 array&lt;float&gt; 报错**：加上 `--embedding-column`。
- **表已存在**：默认报错退出。确认要换数据时加 `--overwrite`。
- **parquet 缺 id**：直接报错。先用 `convert` 从 fvecs 生成带 id 的 parquet，或自己补一列整数 id。
