#!/bin/bash

# =============================================================================
# Dataflow Agent Service Startup Script with Configuration
# DFA服务启动脚本（包含配置）
# =============================================================================

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Configuration Section
# 配置部分
# =============================================================================

load_default_config() {
    print_info "Loading default configuration..."
    print_info "正在加载默认配置..."

    # -----------------------------------------------------------------------------
    # API Configuration (Required)
    # API配置（必需）
    # -----------------------------------------------------------------------------
    export DF_API_KEY="${DF_API_KEY:-your_api_key_here}"                    # API密钥
    export DF_API_URL="${DF_API_URL:-your_api_url_here}"                    # API地址

    # -----------------------------------------------------------------------------
    # Model Configuration
    # 模型配置
    # -----------------------------------------------------------------------------
    export USE_LOCAL_MODEL="${USE_LOCAL_MODEL:-true}"                       # 是否使用本地模型
    export LOCAL_MODEL_NAME_OR_PATH="${LOCAL_MODEL_NAME_OR_PATH:-/mnt/public/model/huggingface/Qwen2.5-7B-Instruct}"  # 本地模型路径

    # -----------------------------------------------------------------------------
    # Server Configuration
    # 服务器配置
    # -----------------------------------------------------------------------------
    export SERVER_HOST="${SERVER_HOST:-0.0.0.0}"                           # 服务器监听地址
    export SERVER_PORT="${SERVER_PORT:-8000}"                              # 服务器端口
    export SERVER_RELOAD="${SERVER_RELOAD:-true}"                          # 是否启用热重载

    # -----------------------------------------------------------------------------
    # Pipeline Configuration
    # 流水线配置
    # -----------------------------------------------------------------------------
    export PIPELINE_JSON_FILE="${PIPELINE_JSON_FILE:-dataflow/example/ReasoningPipeline/pipeline_math_short.json}"  # 流水线JSON配置文件
    export PIPELINE_PY_PATH="${PIPELINE_PY_PATH:-test/recommend_pipeline.py}"                                       # 流水线Python文件路径
    export EXECUTE_THE_PIPELINE="${EXECUTE_THE_PIPELINE:-false}"                                                    # 是否执行流水线

    # -----------------------------------------------------------------------------
    # Operator Configuration
    # 算子配置
    # -----------------------------------------------------------------------------
    export OPERATOR_JSON_FILE="${OPERATOR_JSON_FILE:-dataflow/example/ReasoningPipeline/pipeline_math_short.json}"  # 算子JSON配置文件
    export OPERATOR_PY_PATH="${OPERATOR_PY_PATH:-test/default_op.py}"                                              # 算子Python文件路径
    export EXECUTE_THE_OPERATOR="${EXECUTE_THE_OPERATOR:-false}"                                                    # 是否执行算子

    # -----------------------------------------------------------------------------
    # Timeout and Debug Configuration
    # 超时和调试配置
    # -----------------------------------------------------------------------------
    export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"                      # 请求超时时间（秒）
    export MAX_DEBUG_ROUND="${MAX_DEBUG_ROUND:-5}"                         # 最大调试轮数

    # -----------------------------------------------------------------------------
    # Chat Agent Configuration
    # 聊天代理配置
    # -----------------------------------------------------------------------------
    export DEFAULT_LANGUAGE="${DEFAULT_LANGUAGE:-zh}"                      # 默认语言 zh / en
    export DEFAULT_MODEL="${DEFAULT_MODEL:-gpt-4o-mini}"                   # 默认模型名称
    export DEFAULT_SESSION_KEY="${DEFAULT_SESSION_KEY:-dataflow_demo}"     # 默认会话键

    # -----------------------------------------------------------------------------
    # Test Target Configuration
    # 测试目标配置
    # -----------------------------------------------------------------------------
    export TEST_RECOMMEND_TARGET="${TEST_RECOMMEND_TARGET:-帮我针对数据推荐一个的pipeline!!!不需要去重的算子 ！}"  # 推荐测试目标
    export TEST_WRITE_TARGET="${TEST_WRITE_TARGET:-我需要一个算子，直接使用llm_serving，实现语言翻译，把英文翻译成中文！}"      # 编写测试目标

    # -----------------------------------------------------------------------------
    # Debug Mode (for /config endpoint access)
    # 调试模式（用于访问/config接口）
    # -----------------------------------------------------------------------------
    export DEBUG_MODE="${DEBUG_MODE:-false}"                               # 是否启用调试模式
}

# 配置验证
validate_config() {
    local errors=0
    
    print_info "Validating configuration..."
    print_info "正在验证配置..."
    
    # Check if model path exists when using local model
    # 检查使用本地模型时路径是否存在
    if [ "$USE_LOCAL_MODEL" = "true" ] && [ ! -d "$LOCAL_MODEL_NAME_OR_PATH" ]; then
        print_warning "Local model path does not exist: $LOCAL_MODEL_NAME_OR_PATH"
        print_warning "本地模型路径不存在：$LOCAL_MODEL_NAME_OR_PATH"
    fi
    
    # Check port range
    # 检查端口范围
    if ! [[ "$SERVER_PORT" =~ ^[0-9]+$ ]] || [ "$SERVER_PORT" -lt 1 ] || [ "$SERVER_PORT" -gt 65535 ]; then
        print_error "Invalid SERVER_PORT: $SERVER_PORT"
        print_error "无效的服务器端口：$SERVER_PORT"
        errors=$((errors + 1))
    fi
    
    # Check timeout
    # 检查超时配置
    if ! [[ "$REQUEST_TIMEOUT" =~ ^[0-9]+$ ]] || [ "$REQUEST_TIMEOUT" -lt 1 ]; then
        print_error "Invalid REQUEST_TIMEOUT: $REQUEST_TIMEOUT"
        print_error "无效的请求超时时间：$REQUEST_TIMEOUT"
        errors=$((errors + 1))
    fi
    
    # Check max debug round
    # 检查最大调试轮数
    if ! [[ "$MAX_DEBUG_ROUND" =~ ^[0-9]+$ ]] || [ "$MAX_DEBUG_ROUND" -lt 1 ]; then
        print_error "Invalid MAX_DEBUG_ROUND: $MAX_DEBUG_ROUND"
        print_error "无效的最大调试轮数：$MAX_DEBUG_ROUND"
        errors=$((errors + 1))
    fi
    
    if [ $errors -eq 0 ]; then
        print_success "Configuration validated successfully!"
        print_success "配置验证成功！"
    fi
    
    return $errors
}

# 显示配置摘要
show_config_summary() {
    if [ "$DEBUG_MODE" = "true" ]; then
        echo ""
        echo "=== Configuration Summary ==="
        echo "=== 配置摘要 ==="
        echo "API Key: ${DF_API_KEY:0:8}..."
        echo "API密钥：${DF_API_KEY:0:8}..."
        echo "Server: $SERVER_HOST:$SERVER_PORT"
        echo "服务器：$SERVER_HOST:$SERVER_PORT"
        echo "Use Local Model: $USE_LOCAL_MODEL"
        echo "使用本地模型：$USE_LOCAL_MODEL"
        echo "Model Path: $LOCAL_MODEL_NAME_OR_PATH"
        echo "模型路径：$LOCAL_MODEL_NAME_OR_PATH"
        echo "Debug Mode: $DEBUG_MODE"
        echo "调试模式：$DEBUG_MODE"
        echo "============================"
        echo ""
    fi
}

# =============================================================================
# Startup Script Section
# 启动脚本部分
# =============================================================================

show_usage() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo "用法: $0 [模式] [选项]"
    echo ""
    echo "Modes / 模式:"
    echo "  server      - Start FastAPI server (default) / 启动FastAPI服务器（默认）"
    echo "  recommend   - Run pipeline recommendation test / 运行流水线推荐测试"
    echo "  write       - Run operator writing test / 运行算子编写测试"
    echo "  config      - Show current configuration / 显示当前配置"
    echo "  help        - Show this help message / 显示帮助信息"
    echo ""
    echo "Options / 选项:"
    echo "  --config    - Load custom config file / 加载自定义配置文件"
    echo "  --debug     - Enable debug mode / 启用调试模式"
    echo "  --port      - Set custom port / 设置自定义端口"
    echo ""
    echo "Examples / 示例:"
    echo "  $0 server --port 8080"
    echo "  $0 --config custom_config.sh"
    echo "  $0 recommend --debug"
}

# 解析命令行参数
parse_args() {
    MODE="server"
    CONFIG_FILE=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            server|recommend|write|config|help)
                MODE="$1"
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --debug)
                export DEBUG_MODE="true"
                shift
                ;;
            --port)
                export SERVER_PORT="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                print_error "未知选项: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# 加载配置
load_config() {
    # 首先加载默认配置
    load_default_config
    
    # 如果指定了外部配置文件，则加载它
    if [ -n "$CONFIG_FILE" ]; then
        if [ -f "$CONFIG_FILE" ]; then
            print_info "Loading external configuration from $CONFIG_FILE"
            print_info "正在从 $CONFIG_FILE 加载外部配置"
            source "$CONFIG_FILE"
        else
            print_warning "Config file $CONFIG_FILE not found, using default configuration"
            print_warning "配置文件 $CONFIG_FILE 未找到，使用默认配置"
        fi
    fi
    
    # 验证配置
    if ! validate_config; then
        print_error "Configuration validation failed!"
        print_error "配置验证失败！"
        exit 1
    fi
    
    # 显示配置摘要
    show_config_summary
}

# 环境检查
check_environment() {
    print_info "Checking environment..."
    print_info "正在检查环境..."
    
    # Python检查
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed"
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 依赖检查
    python3 -c "import fastapi, dataflow" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "Missing required packages (fastapi, dataflow)"
        print_error "缺少必需的包 (fastapi, dataflow)"
        exit 1
    fi
    
    print_success "Environment check passed"
    print_success "环境检查通过"
}

# 启动服务
start_service() {
    case $MODE in
        "server")
            print_info "Starting FastAPI server..."
            print_info "正在启动FastAPI服务器..."
            print_info "Server will be available at http://${SERVER_HOST}:${SERVER_PORT}"
            print_info "服务器将在 http://${SERVER_HOST}:${SERVER_PORT} 可用"
            python3 run_dataflow_agent_service.py server
            ;;
        "recommend")
            print_info "Running pipeline recommendation test..."
            print_info "正在运行流水线推荐测试..."
            python3 run_dataflow_agent_service.py recommend
            ;;
        "write")
            print_info "Running operator writing test..."
            print_info "正在运行算子编写测试..."
            python3 run_dataflow_agent_service.py write
            ;;
        "config")
            print_info "Showing current configuration..."
            print_info "正在显示当前配置..."
            python3 run_dataflow_agent_service.py config
            ;;
        "help")
            show_usage
            ;;
    esac
}

# 主函数
main() {
    print_info "Dataflow Agent Service Manager"
    print_info "DFA服务管理器"
    
    parse_args "$@"
    load_config
    check_environment
    start_service
}

# 信号处理
trap 'print_info "Shutting down..."; print_info "正在关闭..."; exit 0' INT TERM

# 执行主函数
main "$@"