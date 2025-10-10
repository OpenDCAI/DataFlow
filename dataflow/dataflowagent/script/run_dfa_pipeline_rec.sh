#!/usr/bin/env bash
# ================================================================
# run_dataflow.sh
# 作用：解析参数 → 导出 DF_API_KEY → 调用 run_dataflow.py
# ./run_dataflow.sh \
#   --json /data/full_dataset.jsonl \
#   --pipeline ./tests/my_pipeline.py \
#   --target "推荐pipeline" \
#   --model gpt-4o-mini \
#   --language zh \
#   --no-debug
# ================================================================
set -euo pipefail

# ——— 默认值（可按需修改） ———
JSON_FILE="`pwd`/data/mq_test_data.jsonl"
PIPELINE_PY="`pwd`/my_pipeline.py"
API_URL="http://123.129.219.111:3000/v1/"
API_KEY="sk-dummy"
MODEL="gpt-4o"
LANGUAGE="en"
TARGET="我需要 2 个reasoning的算子！"
NEED_DEBUG=true
DEBUG_ROUNDS=3

usage() {
  grep '^#' "$0" | cut -c 4-
  exit 0
}

# ——— 解析参数 ———
while [[ $# -gt 0 ]]; do
  case "$1" in
    --json|-j)          JSON_FILE="$2";    shift 2 ;;
    --pipeline|-p)      PIPELINE_PY="$2";  shift 2 ;;
    --url|-u)           API_URL="$2";      shift 2 ;;
    --key|-k)           API_KEY="$2";      shift 2 ;;
    --model|-m)         MODEL="$2";        shift 2 ;;
    --language|-l)      LANGUAGE="$2";     shift 2 ;;
    --target|-t)        TARGET="$2";       shift 2 ;;
    --debug)            NEED_DEBUG=true;   shift ;;
    --no-debug)         NEED_DEBUG=false;  shift ;;
    --debug-rounds|-r)  DEBUG_ROUNDS="$2"; shift 2 ;;
    --help|-h)          usage ;;
    *)  echo "未知参数 $1 ；--help 查看用法" ; exit 1 ;;
  esac
done

# ——— 环境变量 ———
export DF_API_KEY="$API_KEY"

# ——— 执行 Python ———
python run_dfa_pipeline_rec.py \
  --json-file     "$JSON_FILE" \
  --pipeline-file "$PIPELINE_PY" \
  --chat-api-url  "$API_URL" \
  --model         "$MODEL" \
  --language      "$LANGUAGE" \
  --target        "$TARGET" \
  $( $NEED_DEBUG && echo "--debug" || echo "--no-debug" ) \
  --debug-rounds  "$DEBUG_ROUNDS"