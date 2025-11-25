#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTTPSTAN_VERSION="4.13.0"
PYTHON_VERSION="3.10.13"
VENV_NAME="pystan_env"
BUILD_DIR="${PROJECT_DIR}/build"
HTTPSTAN_SOURCE_DIR="${BUILD_DIR}/httpstan-${HTTPSTAN_VERSION}"

echo "=== Installing httpstan and pystan on Mac ARM64 ==="
echo "Project directory: ${PROJECT_DIR}"
echo ""

check_homebrew() {
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        echo "✓ Homebrew is already installed"
    fi
}

install_stanc3() {
    if command -v stanc &> /dev/null; then
        echo "✓ stanc3 is already installed"
        stanc --version
    else
        echo "Installing stanc3 via Homebrew..."
        brew install stanc3
        echo "✓ stanc3 installed"
    fi
}

setup_python_env() {
    if [ -d "${PROJECT_DIR}/${VENV_NAME}" ]; then
        echo "✓ Virtual environment already exists at ${PROJECT_DIR}/${VENV_NAME}"
    else
        echo "Creating Python virtual environment..."
        if command -v pyenv &> /dev/null; then
            pyenv local ${PYTHON_VERSION} || echo "Note: pyenv local failed, using system Python"
        fi
        python3 -m venv "${PROJECT_DIR}/${VENV_NAME}"
        echo "✓ Virtual environment created"
    fi
    
    source "${PROJECT_DIR}/${VENV_NAME}/bin/activate"
    echo "✓ Virtual environment activated"
}

download_httpstan() {
    mkdir -p "${BUILD_DIR}"
    
    if [ -d "${HTTPSTAN_SOURCE_DIR}" ]; then
        echo "✓ httpstan source already exists at ${HTTPSTAN_SOURCE_DIR}"
    else
        echo "Downloading httpstan ${HTTPSTAN_VERSION} from GitHub..."
        cd "${BUILD_DIR}"
        curl -L -o "httpstan-${HTTPSTAN_VERSION}.tar.gz" \
            "https://github.com/stan-dev/httpstan/archive/${HTTPSTAN_VERSION}.tar.gz"
        tar -xzf "httpstan-${HTTPSTAN_VERSION}.tar.gz"
        echo "✓ httpstan downloaded and extracted"
    fi
}

build_httpstan() {
    echo "Building httpstan shared libraries..."
    echo "This may take 10-20 minutes..."
    cd "${HTTPSTAN_SOURCE_DIR}"
    
    if [ -f "httpstan/stanc" ]; then
        echo "✓ httpstan already built (stanc binary exists)"
    else
        make
        echo "✓ httpstan build completed"
    fi
}

replace_stanc() {
    echo "Replacing x86_64 stanc with ARM64 version..."
    cd "${HTTPSTAN_SOURCE_DIR}"
    
    if [ -f "httpstan/stanc" ]; then
        STANC_ARCH=$(file httpstan/stanc | grep -o "arm64\|x86_64" || echo "unknown")
        if [ "$STANC_ARCH" = "arm64" ]; then
            echo "✓ stanc is already ARM64"
        else
            echo "Backing up x86_64 stanc..."
            cp httpstan/stanc httpstan/stanc.x86_64.backup
            
            echo "Replacing with ARM64 version from Homebrew..."
            cp $(which stanc) httpstan/stanc
            chmod +x httpstan/stanc
            
            echo "Verifying ARM64 architecture..."
            file httpstan/stanc
            echo "✓ stanc replaced with ARM64 version"
        fi
    else
        echo "Error: httpstan/stanc not found. Build may have failed."
        exit 1
    fi
}

install_httpstan() {
    source "${PROJECT_DIR}/${VENV_NAME}/bin/activate"
    
    if python -c "import httpstan" 2>/dev/null; then
        echo "✓ httpstan is already installed"
    else
        echo "Installing httpstan from source..."
        pip install "${HTTPSTAN_SOURCE_DIR}"
        echo "✓ httpstan installed"
    fi
}

install_pystan() {
    source "${PROJECT_DIR}/${VENV_NAME}/bin/activate"
    
    if python -c "import stan" 2>/dev/null; then
        echo "✓ pystan is already installed"
    else
        echo "Installing pystan..."
        pip install pystan
        echo "✓ pystan installed"
    fi
}

install_nest_asyncio() {
    source "${PROJECT_DIR}/${VENV_NAME}/bin/activate"
    
    if python -c "import nest_asyncio" 2>/dev/null; then
        echo "✓ nest_asyncio is already installed"
    else
        echo "Installing nest_asyncio..."
        pip install nest_asyncio
        echo "✓ nest_asyncio installed"
    fi
}

verify_installation() {
    source "${PROJECT_DIR}/${VENV_NAME}/bin/activate"
    
    echo ""
    echo "=== Verifying Installation ==="
    
    python << EOF
import stan
import httpstan
import nest_asyncio

nest_asyncio.apply()

print(f"✓ stan version: {stan.__version__}")
print(f"✓ httpstan version: {httpstan.__version__}")

model_code = """
parameters {
    real y;
}
model {
    y ~ normal(0, 1);
}
"""

model = stan.build(model_code)
print("✓ Model compilation successful!")
print("")
print("Installation verified successfully!")
EOF
}

main() {
    check_homebrew
    install_stanc3
    setup_python_env
    download_httpstan
    build_httpstan
    replace_stanc
    install_httpstan
    install_pystan
    install_nest_asyncio
    verify_installation
    
    echo ""
    echo "=== Installation Complete ==="
    echo "To activate the environment, run:"
    echo "  source ${PROJECT_DIR}/${VENV_NAME}/bin/activate"
}

main


