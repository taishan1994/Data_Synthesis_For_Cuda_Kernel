set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

pip install -r "${ROOT_DIR}/requirements.txt" --user -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install pydantic-settings --user -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
apt update
apt-get install iproute2 redis -y