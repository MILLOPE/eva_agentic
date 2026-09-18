# eva_agentic

`eva_agentic` 用于调用各个框架原有的评测命令，并为实验保留可追溯的记录。它不会替代框架自身的规划器、模型服务、模拟器或控制循环。

每个框架继续使用自己的 conda、uv 或容器环境。评测器在仓库中保存一份可移植的框架声明文件，同时在版本控制之外保存一份本地配置。这个本地配置用于提供 conda 或 uv 可执行文件、环境名称、模型与资源路径、GPU 槽位以及密钥等机器相关信息。

干净的框架源码、基准测试源码和模型服务源码统一放在 [`third_party/`](third_party/README.md) 下。基准测试目录是共享的。例如，所有使用 LIBERO 的框架都共用 `third_party/benchmarks/libero`。如果某个框架需要做特定修改，应以小型补丁的形式记录，而不是复制出一份新的 benchmark 目录。

评测器自身使用项目内的 `.venv/`。所有框架原生环境或模型环境都应放在 `.envs/eva-*` 下，该目录已被 Git 忽略。这样可以让实验依赖保持在项目内部，同时避免修改共享的 conda 环境。

可以从 [RATs 框架声明示例](examples/rats-libero.frameworks.example.yaml) 和 [本地配置示例](examples/local-profile.example.yaml) 开始。框架声明文件可以安全提交到版本库；本地配置则应先复制到仓库外或被 Git 忽略的位置，再填写路径、密钥和机器相关参数。

## 工作流程

使用实验配置和 JSONL case 列表创建一次实验，并将其固定下来：

```bash
eva-agentic init --experiment experiment.json --runs-root runs --run-id smoke
```

使用版本化的框架声明和未跟踪的本地配置，运行框架原生评测任务：

```bash
eva-agentic run \
  --run-dir runs/smoke \
  --frameworks frameworks.json \
  --profile local-profile.yaml
```

查看标准化后的结果汇总：

```bash
eva-agentic summary --run-dir runs/smoke
```

每次尝试都会记录解析后的实际命令、工作目录、分配的资源、时间戳、退出码、日志以及框架原生输出文件。

需要注意：**退出码为 0 并不代表任务成功。** 框架适配器必须解析框架自己的结果文件，并明确记录它所采用的成功判定信号。

## 常用命令

以下命令均应在仓库根目录执行。只需要激活评测器自身的环境；运行器会根据本地配置，为不同框架选择各自的原生解释器。

```bash
cd /share/repos/eva_agentic
source .venv/bin/activate
eva-agentic --help
```

根据实验配置创建新的运行目录。每次实验都应使用新的运行 ID，从而保证历史实验产生的证据不会被覆盖或修改。

```bash
RUN_ID="rpent-vllm-user-libero-$(date -u +%Y%m%d-%H%M%S)"

eva-agentic init \
  --experiment examples/rpent-vllm-user-libero.experiment.example.yaml \
  --runs-root runs \
  --run-id "$RUN_ID"
```

首次使用时，准备一份机器本地配置，然后填写解释器、服务端点、GPU 槽位、模型参数和签名信息。`profiles/` 目录已被 Git 忽略。

```bash
cp examples/rpent-vllm-user-libero.profile.example.yaml \
  profiles/rpent-vllm-user-libero.yaml

$EDITOR profiles/rpent-vllm-user-libero.yaml
```

通过评测器运行框架原生命令。RPent vllm-user 示例默认使用签名 vLLM 服务；开始运行前需要已有 VLA/SAM3 服务，并通过 `scripts/with-rpent-vllm-user-env.sh` 注入签名变量。

```bash
RUN_DIR="runs/$RUN_ID"

scripts/with-rpent-vllm-user-env.sh ./.venv/bin/eva-agentic run \
  --run-dir "$RUN_DIR" \
  --frameworks examples/rpent-vllm-user-libero.frameworks.example.yaml \
  --profile profiles/rpent-vllm-user-libero.yaml
```

查看标准化后的结果，并检查保留下来的实验文件：

```bash
eva-agentic summary --run-dir "$RUN_DIR"

find "$RUN_DIR" -maxdepth 3 -type f | sort
```

如果只想检查评测器和 RPent 的入口是否能够正常启动，而不实际运行 episode，可以执行：

```bash
./.venv/bin/eva-agentic --help

ENV="$PWD/.envs/eva-rpent-vllm-user"

"$ENV/bin/python" -m rpent.cli.main --help
```

修改适配器、配置加载逻辑或结果解析器之后，应运行项目测试：

```bash
./.venv/bin/python -m pytest -q
```

VLA 和 SAM3 进程暴露的是 RPent JSON-RPC 接口。它们的 `healthz` 方法需要通过 `/call` 调用；直接向 `/health` 发起普通 HTTP GET 请求并不是有效的健康检查方式。

```bash
for port in 18114 18115; do
  curl -fsS -H 'Content-Type: application/json' \
    -d '{"method":"healthz","args":[],"kwargs":{}}' \
    "http://127.0.0.1:${port}/call"
  echo
done
```

## RPent 与共享 LIBERO

`rpent_libero` 使用统一的 `third_party/benchmarks/libero` checkout。

它会启动 RPent 原生的 LIBERO 环境服务，并通过未跟踪的本地配置中的 `RPENT_VLA_ENDPOINT` 和 `RPENT_SAM3_ENDPOINT` 连接 VLA 与 SAM3 服务。

评测结果从 RPent 原生输出中的以下文件读取：

```text
{suite-without-libero_}_t{task}_s{seed}.json
```

其中顶层布尔字段 `terminated` 被作为 benchmark 的任务成功信号。

本地配置必须提供 VLA 和 SAM3 两个服务端点；同时，每个 case 都必须明确指定以下字段：

* `suite`
* 整数类型的 `task_id`
* `seed`
* `max_episode_steps`

可以使用以下三个示例文件作为最小 smoke 配置：

* [框架声明](examples/rpent-vllm-user-libero.frameworks.example.yaml)
* [单 case 列表](examples/rpent-vllm-user-libero.cases.example.jsonl)
* [实验配置](examples/rpent-vllm-user-libero.experiment.example.yaml)

这些示例只用于进行小规模 smoke 检查，并不代表已经完成真实机器环境下的完整验证。

示例默认使用 vllm-user 的签名 vLLM 服务。签名变量由
`scripts/with-rpent-vllm-user-env.sh` 从本机 RPent 环境注入，不写入仓库或
framework 声明；VLA/SAM3 端点放在被 Git 忽略的本地配置中。

当前 eva 使用的 RPent baseline 不注册 `vllm_user` planner；本节的最小 registration patch 和可替换 planner snapshot 会补上这个 native 入口。

### RPent vllm_user planner smoke run

当前 eva 的 vllm_user 接入使用本仓库内的
third_party/frameworks/rpent，planner snapshot 和最小 RPent registration
补丁位于 patches/rpent/libero/vllm_user。/share/repos/Rpent-correction
只作为迁移来源，不会进入运行时 PYTHONPATH。planner 可以在后续手动替换
snapshot，但旧 run 目录不会被覆盖。

本机已有的 /share/repos/Rpent-correction/.env.local 同时包含 correction 专用 PYTHONPATH、旧解释器和 vLLM/VLA/SAM3 配置，不能直接 source 后运行 eva。运行时请使用 scripts/with-rpent-vllm-user-env.sh；它只保留认证和服务连接变量，清除旧 Python 路径，并强制 RPENT_REPO_ROOT 和 VIRTUAL_ENV 指向当前 eva。

如果当前机器没有 CUDA，应在 GPU 主机上执行下面的 eva 命令，或把远端 VLA/SAM3 endpoint 通过 SSH 转发到 profile 中的地址；不要在本机启动 GPU 服务。

先应用当前本地 RPent 的补丁，并在 dedicated environment 安装 RPent 和
planner：

    cd /share/repos/eva_agentic
    patch -p1 < patches/rpent/libero/vllm_user/rpent-vllm-user.patch

    ENV="$PWD/.envs/eva-rpent-vllm-user"
    UV=/share/bin/uv-x86_64-unknown-linux-gnu/uv
    "$UV" pip install --python "$ENV/bin/python" -e third_party/frameworks/rpent
    "$UV" pip install --python "$ENV/bin/python" -e patches/rpent/libero/vllm_user/planner

复制并填写只存在于本机的 profile。RPENT_VLLM_PRIVATE_KEY 和
RPENT_VLLM_SIGNER_URL 必须二选一；它们的内容不提交。

    cp examples/rpent-vllm-user-libero.profile.example.yaml \
      profiles/rpent-vllm-user-libero.yaml
    $EDITOR profiles/rpent-vllm-user-libero.yaml

运行 smoke 前依次检查 planner 来源、native CLI、signed vLLM、VLA 和
SAM3。VLA/SAM3 使用 RPent JSON-RPC 的 /call healthz，而不是普通
/health：

    scripts/with-rpent-vllm-user-env.sh "$ENV/bin/python" -c \
      "import rpent, rpent_vllm_user; print(rpent.__file__); print(rpent_vllm_user.__file__)"
    scripts/with-rpent-vllm-user-env.sh "$ENV/bin/rpent" --help
    scripts/with-rpent-vllm-user-env.sh \
      "$ENV/bin/rpent-check-llm" --planner vllm_user --model default --json

    for port in 18114 18115; do
      curl -fsS -H 'Content-Type: application/json' \
        -d '{"method":"healthz","args":[],"kwargs":{}}' \
        "http://127.0.0.1:$port/call"
      echo
    done

使用唯一的 run id 初始化和运行一个固定 case：

    RUN_ID="rpent-vllm-user-libero-$(date -u +%Y%m%d-%H%M%S)"

    scripts/with-rpent-vllm-user-env.sh ./.venv/bin/eva-agentic init \
      --experiment examples/rpent-vllm-user-libero.experiment.example.yaml \
      --runs-root runs \
      --run-id "$RUN_ID"

    scripts/with-rpent-vllm-user-env.sh ./.venv/bin/eva-agentic run \
      --run-dir "runs/$RUN_ID" \
      --frameworks examples/rpent-vllm-user-libero.frameworks.example.yaml \
      --profile profiles/rpent-vllm-user-libero.yaml

    ./.venv/bin/eva-agentic summary --run-dir "runs/$RUN_ID"
    find "runs/$RUN_ID" -maxdepth 6 -type f | sort

成功判定只读取 RPent 原生 audit 的布尔字段 terminated == true。
launch.json 保存实际命令、工作目录、资源和非敏感 provenance；
stdout.log、stderr.log、RPent transcript、audit 和 recipe 都保留在 attempt
目录中。退出码为 0 或 planner 的 finish_result 都不替代该环境成功信号。

### 大规模评测：容量探针

扩容前先量化瓶颈（远端 vLLM/SAM3 吞吐、本地 CPU/EGL 仿真、evidence
体积），不要直接加大并发。工具都在 `scripts/` 下，纯逻辑在
`src/eva_agentic/probing.py`，均有单测覆盖。统一通过 `eva-agentic` 这个
入口调用：`gen-cases` 生成矩阵，`probe` 跑探针。

先生成固定 case 矩阵（task × seed，避免只用一个种子）：

    ./.venv/bin/eva-agentic gen-cases \
      --suite libero_object --tasks 0,1,4-6 \
      --seed-base 0 --seed-count 3 \
      --max-episode-steps 500 --out /tmp/libero-matrix.jsonl

capacity probe 把同一 case 列表以不同并发档各跑成一个独立 run（新 run
id、各自渲染的 experiment/profile，attempt 互不覆盖），并输出每档
success/耗时对比：

    scripts/with-rpent-vllm-user-env.sh ./.venv/bin/eva-agentic probe \
      --experiment examples/rpent-vllm-user-libero.probe.experiment.yaml \
      --cases /tmp/libero-matrix.jsonl \
      --frameworks examples/rpent-vllm-user-libero.frameworks.example.yaml \
      --profile profiles/rpent-vllm-user-libero.yaml \
      --concurrency 1,2,4,8 --tag capacity --out runs/probe-capacity.csv

CSV 里的 planned/completed/success/task_failure/invalid/infra 与
ep_median_s/ep_max_s 区分任务失败与基础设施失败；成功判定仍只来自
RPent audit 的 terminated。探针属于 exploration，不用于正式 evaluation。

- `--concurrency` 同时充当 `execution.max_jobs` 与 resource slot 数量
  （slots = 并发数 × 每任务需求）。

结果可视化：`visualize` 聚合 `runs/` 下所有冻结 run 成一个 per-case
summary（CSV/JSON 为主审计产物），并用纯 Python 生成 SVG 图与一个
HTML 报告（不引入 matplotlib 等重量依赖）：

    ./.venv/bin/eva-agentic visualize --runs-root runs --out-dir runs/.probe/viz

输出含 `summary.csv`、`summary.json`、`report.html`，以及
`success_rate.svg`（按任务成功率）、`status_heatmap.svg`（task × seed
状态）、`duration.svg`（单条耗时，log scale）。聚合逻辑在
`src/eva_agentic/viz.py`，只读、不覆盖历史 run。
- 先离线验证配置渲染：加 `--dry-run` 只渲染不跑服务。
- 定正式规模前先确定“某并发下成功率不下降、不出现超时误判”的安全并发
  数，并核对当前 slots 上限（默认 `gpu: [0]` 会把并发锁成 1）。
- 关注 evidence 体积：smoke 是 3 步约 28MB，几百步的单集是 GB 级；正式
  矩阵前先确定留存策略与 artifacts 落点。

### 在项目内创建 RPent 环境

使用 uv 在当前项目内部创建 RPent 环境：

```bash
ROOT="$PWD"

UV=/share/bin/uv-x86_64-unknown-linux-gnu/uv

ENV="$ROOT/.envs/eva-rpent-vllm-user"

mkdir -p "$ROOT/.envs"

"$UV" venv --python 3.11 "$ENV"

"$UV" pip install --python "$ENV/bin/python" --no-deps \
  -e "$ROOT/third_party/frameworks/rpent"

"$UV" pip install --python "$ENV/bin/python" --prerelease=allow \
  "pydantic-ai-slim[anthropic,openai]>=2.1" "pydantic>=2" \
  "fastapi>=0.110" "uvicorn>=0.27" "httpx>=0.27" \
  "jsonschema>=4.18" "mcp>=1.23.0,<2.0.0" \
  "huggingface_hub>=0.24" "prompt-toolkit>=3.0.48" \
  numpy scipy imageio "PyYAML>=6.0"

"$UV" pip install --python "$ENV/bin/python" --no-deps \
  "rpent-rlinf==0.3.0" \
  "$ROOT/third_party/benchmarks/libero"

"$UV" pip install --python "$ENV/bin/python" --no-deps \
  "robosuite==1.5.2" "bddl==3.6.0" "mujoco==3.3.0" \
  termcolor easydict cloudpickle
```

第一条安装命令只安装 RPent 的 CLI，不安装可选的模型相关依赖。

第二组依赖用于补充 planner 和 API 层。

最后两条安装命令则安装 RLinf，以及共享 LIBERO checkout 所需要的模拟器版本。

如果机器环境完全独立、不需要担心影
