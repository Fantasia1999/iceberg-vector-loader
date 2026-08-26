# iceberg-vector-loader

- 把本地向量数据导入 **Iceberg v3** 表。
- 写入走 **Spark 原生**（`spark-sql` + Hadoop catalog），不是 pyiceberg。
- 同一份 parquet / fvecs / ivecs 也可以转成 **Lance** dataset，方便本地做 KNN / ANN 查询。
- 适用场景：SIFT1M / SIFT10M 以及其他本地 embedding 数据集。namespace、表名都可以指定。

| 组件 | 版本 |
|---|---|
| Spark | 3.5.9-bin-hadoop3 |
| Iceberg | `iceberg-spark-runtime-3.5_2.12-1.11.0.jar`（已随仓库提交） |
| JDK | 21 |
| Python | ≥ 3.11 且 < 3.15（3.11 / 3.12 / 3.13 / 3.14） |

入口：`python -m iceberg_vector_loader <命令>`，或 `iceberg-vector-loader <命令>`。

| 命令 | 作用 | 是否需要 JDK / Spark |
|---|---|---|
| `convert` | 只把 `.fvecs` / `.ivecs` 转成 parquet | 否 |
| `to-lance` | 把 parquet / fvecs / ivecs 转成 Lance dataset | 否 |
| `query-lance` | 在 Lance dataset 上做 KNN / SQL 查询 | 否 |
| `load` | 准备输入并写入 Iceberg v3 表 | 是 |
| `bootstrap` | 解析 / 下载 JDK 21 和 Spark 3.5.9 | 下载时需要网络 |

## 架构

```
.fvecs / .ivecs                 .parquet / 目录 / glob
   │                                    │
   ▼                                    │
convert（parquet 中转）                 │
   │                                    │
   ├──────────────┬─────────────────────┤
                  │                     │
                  ▼                     ▼
            load：写入 Iceberg     to-lance：写出 Lance
            spark-sql + v3         定长 list（embedding / neighbors）
                  │                     │
                  ▼                     ▼
            Hadoop catalog         query-lance：KNN / SQL
            <warehouse>/...        <name>.lance/
```

`load`：Python 只做数据准备，建表由 Spark 的 Iceberg Java writer 完成，因此是满血 V3（含 `next-row-id` 等）。表已存在且未指定 `--overwrite` 时直接退出，不会 append。

`to-lance` / `query-lance`：全程 Python（`pylance`），不经过 Spark。

## 安装

### Python

所有命令都需要：

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### JDK 21 和 Spark 3.5.9

只 `convert` / `to-lance` / `query-lance` 的话到这里就可以了。`load` 还需要 **JDK 21** 和 **Spark 3.5.9-bin-hadoop3**。本机已经有的话不必下载，任选下面一种即可。Iceberg runtime jar 已在仓库里（`third_party/iceberg-spark-runtime-3.5_2.12-1.11.0.jar`），一般不用另配。

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

只把 TexMex `.fvecs` / `.ivecs` 转成 parquet，**不建表、不启动 Spark**。

| 输入 | 写出列 |
|---|---|
| `.fvecs` | `id`（`int64`，默认 `0 .. n-1`）+ `embedding`（`list<float32>`） |
| `.ivecs` | `id` + `neighbors`（`list<int32>`，通常是 ground-truth 近邻 id） |

```bash
python -m iceberg_vector_loader convert \
  --input /path/to/sift_base.fvecs \
  --output /path/to/sift_base.parquet

python -m iceberg_vector_loader convert \
  --input /path/to/sift_groundtruth.ivecs \
  --output /path/to/sift_groundtruth.parquet
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | 必填 | TexMex `.fvecs`（`int32 dim + dim * float32`）或 `.ivecs`（`int32 dim + dim * int32`） |
| `--output` | 必填 | 目标 parquet 路径 |
| `--id-offset` | `0` | 起始 id，写成 `id_offset .. id_offset+n-1` |
| `--batch-size` | `250000` | 每批行数，用来限制转换时的内存 |

`load` 遇到 `.fvecs` 时走同一套转换。`.ivecs` 只给 `convert` / `to-lance`（近邻 id，不是向量），不要拿去 `load`。已经转好的向量 parquet 可以直接给 `load --input` 或 `to-lance --input`。

## to-lance

把 parquet / fvecs / ivecs 转成 Lance dataset，**不启动 Spark**。`.fvecs` / `.ivecs` 先走 `convert`，把 parquet 写到 Lance 输出旁边（`--output ./sift.lance` → `./sift.parquet`），再读这份 parquet 写出 Lance。

- 浮点 embedding（fvecs / 普通向量 parquet）：写成 `fixed_size_list<float32>[dim]`，可 `--index` 后做 KNN / ANN。
- 整数 neighbors（ivecs / ground-truth parquet）：写成 `fixed_size_list<int32>[k]`，不能建向量索引，用 `query-lance --sql` 查看。

列名推断规则和 `load` 相同：`id` + `embedding`；ivecs 则是 `id` + `neighbors`。其余列原样带上。

```bash
python -m iceberg_vector_loader to-lance \
  --input /data/sift10m.parquet \
  --output ./sift10m.lance

# fvecs / ivecs：先在 Lance 输出旁写出 .parquet，再转 Lance
python -m iceberg_vector_loader to-lance \
  --input /path/to/sift_base.fvecs \
  --output ./sift_base.lance \
  --index

python -m iceberg_vector_loader to-lance \
  --input /path/to/sift_groundtruth.ivecs \
  --output ./sift_gt.lance
```

分片 parquet：

```bash
python -m iceberg_vector_loader to-lance \
  --input '/data/bioasq_large_10m/shuffle_train-*-of-10.parquet' \
  --output ./bioasq_train.lance
```

大规模数据建议顺带建 IVF_PQ 索引，后面 `query-lance` 才是 ANN 而不是全表扫描：

```bash
python -m iceberg_vector_loader to-lance \
  --input /data/sift1m.parquet \
  --output ./sift1m.lance \
  --index \
  --metric L2
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | 必填 | `.fvecs` / `.ivecs` / `.parquet` / parquet 目录 / parquet glob（glob 必须加引号） |
| `--output` | 必填 | Lance dataset 目录，通常以 `.lance` 结尾 |
| `--overwrite` | 关 | 目标已存在则重建。不加此参数时已存在直接报错 |
| `--id-column` / `--embedding-column` | 自动推断 | 与 `load` 相同 |
| `--id-offset` | `0` | 仅用于 fvecs / ivecs 转 parquet 时的起始 id |
| `--batch-size` | `250000` | 读 parquet / 转 TexMex 时每批行数 |
| `--index` | 关 | 写完后在 `embedding` 上建向量索引 |
| `--index-type` | `IVF_PQ` | `IVF_PQ` / `IVF_HNSW_PQ` / `IVF_HNSW_SQ` |
| `--metric` | `L2` | `L2` / `cosine` / `dot`。cosine 请自行保证向量已归一化 |
| `--num-partitions` | `sqrt(rows)` | IVF 分区数 |
| `--num-sub-vectors` | `dim/8` | PQ 子向量数；`dim` 能被 8 整除时最快 |

小数据集（几百行）不必 `--index`，brute-force KNN 就够。IVF_PQ 要求行数明显多于分区数，数据太少建索引会失败。

## query-lance

对 `to-lance` 写出的 **底库** dataset 做查询。向量检索必须同时给 `--queries` 和 `--query-id`：query 向量只来自独立 query 集，不能从底库取行。

SIFT：底库 `sift_base.lance`，query 用 `sift_query.parquet`（再和 `sift_groundtruth.ivecs` 对召回）。

```bash
# 用 query 集 id=0 去搜 base
python -m iceberg_vector_loader query-lance \
  --dataset ./sift_base.lance \
  --queries ./sift_query.parquet \
  --query-id 0 \
  --k 10

# 打印完整 query 向量
python -m iceberg_vector_loader query-lance \
  --dataset ./sift_base.lance \
  --queries ./sift_query.parquet \
  --query-id 0 \
  --k 5 \
  --verbose

# 手写 query 向量（逗号分隔或 JSON list）
python -m iceberg_vector_loader query-lance \
  --dataset ./sift10m.lance \
  --query-vector '0.1,0.2,0.3' \
  --k 10

# 从 query 集随机抽一条
python -m iceberg_vector_loader query-lance \
  --dataset ./sift_base.lance \
  --queries ./sift_query.parquet \
  --sample --seed 0 \
  --k 10
```

带 metadata 过滤的向量检索（先过滤再搜）：

```bash
python -m iceberg_vector_loader query-lance \
  --dataset ./sift_base.lance \
  --queries ./sift_query.parquet \
  --query-id 0 \
  --filter 'id < 1000' \
  --k 10
```

已建 IVF 索引时，可用 `--nprobes` / `--refine-factor` 调召回，或 `--no-index` 强制精确 KNN：

```bash
python -m iceberg_vector_loader query-lance \
  --dataset ./sift_base.lance \
  --queries ./sift_query.parquet \
  --query-id 0 \
  --k 10 \
  --nprobes 16 \
  --refine-factor 5
```

普通 SQL（不是向量检索）。ivecs / ground-truth 表只能走这条路径：

```bash
python -m iceberg_vector_loader query-lance \
  --dataset ./sift10m.lance \
  --sql 'SELECT id FROM dataset LIMIT 5'

python -m iceberg_vector_loader query-lance \
  --dataset ./sift_gt.lance \
  --sql 'SELECT id, neighbors FROM dataset LIMIT 5'
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | 必填 | 要搜索的底库 Lance dataset（如 `sift_base.lance`） |
| `--queries` | 与 `--query-id` 同时必填 | 独立 query 集：`.parquet` / `.fvecs` / Lance。SIFT 用 `sift_query.parquet` |
| `--query-id` | 与 `--queries` 同时必填 | query 集里的行 id，取该行 embedding 去搜底库 |
| `--query-vector` | 无 | 手写向量（逗号分隔或 JSON）；与 `--queries` 互斥 |
| `--sample` | 关 | 从 `--queries` 随机抽一行，代替 `--query-id` |
| `--k` | `10` | 近邻个数 |
| `--columns` | 除 `embedding` 外全部 | 逗号分隔；向量检索会额外带 `_distance` |
| `--include-embedding` | 关 | 结果里带上 embedding 列 |
| `--filter` | 无 | SQL 谓词，作为向量检索的 prefilter |
| `--sql` | 无 | 跑 `SELECT ... FROM dataset ...`，不再做 KNN |
| `--metric` | `L2` | 应与建索引时的 metric 一致 |
| `--nprobes` | Lance 默认 | IVF 探测的分区数 |
| `--refine-factor` | 无 | ANN 多取 `k * refine_factor` 再按真实距离重排 |
| `--no-index` | 关 | 忽略已有向量索引，精确扫描 |
| `--verbose` | 关 | 完整打印 query 向量（以及结果里的 list 列）；默认只显示前 8 个元素 |

默认不打印 embedding（太长）。距离列是 `_distance`：L2 时是平方欧氏距离。精确 KNN 下 query 自己命中约为 `0`；IVF_PQ 未 refine 时是近似距离。

### Python 例子

```python
import lance
from iceberg_vector_loader import convert_to_lance, query_lance

convert_to_lance("/data/sift_base.parquet", "./sift_base.lance", overwrite=True)

# 1) 用 query 集 id=0 去搜底库
hits = query_lance("./sift_base.lance", queries_path="./sift_query.parquet", query_id=0, k=5)
print(hits.table.to_pydict())

# 2) 精确 KNN（不用索引）+ 过滤
hits = query_lance(
    "./sift_base.lance",
    queries_path="./sift_query.parquet",
    query_id=0,
    k=10,
    filter_expr="id < 10000",
    use_index=False,
)

# 3) 直接用 pylance：SQL、随机读、向量检索
ds = lance.dataset("./sift.lance")
print(ds.count_rows(), ds.schema)
print(ds.sql("SELECT id FROM dataset LIMIT 5").build().to_batch_records())
print(ds.take([0, 100, 500], columns=["id"]))
print(
    ds.to_table(
        columns=["id", "_distance"],
        nearest={"column": "embedding", "q": [0.1] * ds.schema.field("embedding").type.list_size, "k": 10},
    )
)
```

## load

准备输入并写入 Iceberg v3 表。`--input` 可以是：

| 类型 | 说明 |
|---|---|
| `.fvecs` | 先转成 parquet，再导入 |
| `.parquet` / `.parq` | 直接读 footer 推断列 |
| 目录 | 目录内全部 parquet 当作一份输入（Spark `parquet.\`dir\``）；**所有文件 schema 必须一致** |
| glob | 只读匹配的 parquet，例如 `shuffle_train-*-of-10.parquet`。shell 下必须给 pattern 加引号，见 [分片 parquet / glob](#分片-parquet--glob) |

成功后打印表标识、`format-version`、location、维度、写入行数，以及墙钟耗时（总计 / prepare / spark）。

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

分片 parquet（`shuffle_train-00-of-10.parquet` 这类）用 glob，见下一节。

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
| `--input` | 必填 | `.fvecs` / `.parquet` / parquet 目录 / parquet glob（glob 必须加引号）。不接受 `.ivecs` |
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
- parquet 需要落一份规范化副本时（仅当显式传了 `--output`）：你指定的路径
- 只是列名不同（如 `emb` → `embedding`）：不再重写，Spark `SELECT ... AS` 改名

### 分片 parquet / glob

数据集一大就会拆成多个 parquet，常见名字是 HuggingFace 风格的 `*-NN-of-MM.parquet`。`--input` 接受 glob，只读匹配到的文件。

```bash
python -m iceberg_vector_loader load \
  --namespace bioasq \
  --table train \
  --input '/data/bioasq_large_10m/shuffle_train-*-of-10.parquet' \
  --warehouse ./warehouse \
  --driver-memory 64g
```

| 写法 | 结果 |
|---|---|
| `'.../shuffle_train-*-of-10.parquet'` | 10 个 train shard 全部导入 |
| `.../shuffle_train-00-of-10.parquet` | 只导入这一个文件；若同目录还有其它 `*-of-10` shard，会打警告 |
| `.../bioasq_large_10m/` | 目录里全部 `.parquet`。train / test / neighbors / labels 混在一起且 schema 不同时会报错 |

规则：

- **shell 下必须给 glob 加引号**（单引号或双引号）。否则 `*` 会被 shell 展开，`--input` 只拿到第一份文件，后面的路径变成多余参数。
- 匹配到的文件必须是 `.parquet` / `.parq`，且 **schema 完全一致**（列名和类型）。不一致时按 schema 分组列出文件名，然后退出。
- 支持 `*`、`?`、`[…]`，以及需要跨目录时的 `**`。只匹配一层文件名时用 `*` 即可，例如 `shuffle_train-*-of-10.parquet`。
- 同目录里常混着 `test.parquet`、`neighbors.parquet`、`scalar_labels.parquet`。要哪一份就写哪一份的 pattern，不要图省事传整个目录。
- 测试集另导一张表：`--input /data/bioasq_large_10m/test.parquet`。
- 列名是 `emb` / `vector` 这类别名时，Spark `SELECT ... AS embedding` 改名，**不会**先把十几 GB 重写成一份新 parquet。只有显式传了 `--output` 才会落规范化副本。

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
- **目录里混了多种 parquet**：会按 schema 分组报错。用 glob 只选一个 split，例如 `'.../shuffle_train-*-of-10.parquet'`。
- **只导进了一个 shard**：确认 `--input` 是 glob 而不是单个 `00-of-10` 文件；shell 下 glob 必须加引号。
