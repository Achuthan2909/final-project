# Installing httpstan and pystan on Mac M3 (ARM64)

This guide documents the installation process for `httpstan` and `pystan` on Apple Silicon Macs (M1/M2/M3). The standard PyPI installation doesn't work on ARM64, so we need to build from source.

## Prerequisites

- macOS with Apple Silicon (ARM64)
- Python 3.10+ (we used 3.10.13 via pyenv)
- C++ compiler: clang ≥10.0 (comes with Xcode Command Line Tools)
- Homebrew (for installing stanc3)

## Step-by-Step Installation

### 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install stanc3 (Stan Compiler)

The Stan compiler is required for building Stan models. Install the ARM64-compatible version via Homebrew:

```bash
brew install stanc3
```

This installs `stanc3` version 2.37.0 (or newer) which is ARM64-compatible.

### 3. Set Up Python Environment

Create a virtual environment using pyenv (or your preferred method):

```bash
cd /path/to/your/project
pyenv local 3.10.13  # or your preferred Python 3.10+ version
python -m venv pystan_env
source pystan_env/bin/activate
```

### 4. Build httpstan from Source

**Important:** httpstan must be downloaded from source and built locally. The PyPI package only provides x86_64 wheels, which don't work natively on ARM64 Macs.

#### 4.1. Download httpstan Source

You have two options to get the source code:

**Option A: Download from GitHub releases**

```bash
cd /path/to/downloads
# Download the latest release (or specific version)
curl -L -o httpstan-4.13.0.tar.gz https://github.com/stan-dev/httpstan/archive/4.13.0.tar.gz
tar -xzf httpstan-4.13.0.tar.gz
cd httpstan-4.13.0
```

**Option B: Clone from GitHub**

```bash
cd /path/to/downloads
git clone https://github.com/stan-dev/httpstan.git
cd httpstan
# Optionally checkout a specific version
git checkout 4.13.0
```

**Note:** We used version 4.13.0, but you can use any recent version. Check the [releases page](https://github.com/stan-dev/httpstan/releases) for available versions.

#### 4.2. Build Shared Libraries

Navigate to the httpstan source directory and build the required shared libraries:

```bash
cd /path/to/httpstan-4.13.0
make
```

This step:
- Downloads Stan Math and Stan source code
- Builds SUNDIALS libraries (for ODE solving)
- Builds TBB libraries (for threading)
- Downloads and sets up header files

**Note:** This process can take 10-20 minutes depending on your system.

#### 4.3. Replace stanc Binary with ARM64 Version

**Why this is necessary:** The `make` command downloads an x86_64 version of `stanc`. While macOS can run x86_64 binaries under Rosetta emulation, this causes two problems:
- Rosetta emulation is slower (2-3x slower than native ARM64)
- httpstan has a 1-second timeout for compilation, and the x86_64 binary running under Rosetta often exceeds this timeout
- Native ARM64 execution is much faster and more reliable

Replace the x86_64 `stanc` with the ARM64 version from Homebrew:

```bash
# Backup the x86_64 version (optional)
cp httpstan/stanc httpstan/stanc.x86_64.backup

# Replace with ARM64 version from Homebrew
cp $(which stanc) httpstan/stanc
chmod +x httpstan/stanc

# Verify it's ARM64
file httpstan/stanc
# Should show: Mach-O 64-bit executable arm64
```

**Note:** Your system is running natively in ARM64 mode (not Rosetta). The x86_64 binary would work under Rosetta emulation, but it's too slow for httpstan's timeout requirements.

#### 4.4. Install httpstan from Source

**Important:** Do NOT install from PyPI (`pip install httpstan`). You must install from the local source directory after building.

Install httpstan from the source directory:

```bash
# Make sure your virtual environment is activated
source /path/to/your/project/pystan_env/bin/activate

# Install from the local source directory (not from PyPI)
pip install /path/to/httpstan-4.13.0
```

This will:
- Build a wheel with all the compiled libraries (SUNDIALS, TBB)
- Include the ARM64 `stanc` compiler we replaced earlier
- Install the package in your virtual environment

**Note:** The path should point to the httpstan source directory (e.g., `/Users/Studies/Downloads/httpstan-4.13.0`).

### 5. Install pystan

With httpstan installed, install pystan:

```bash
pip install pystan
```

**Note:** The package is installed as `pystan`, but you import it as `stan` in Python.

### 6. Verify Installation

Test that everything works:

```python
import stan
import httpstan

print(f"stan version: {stan.__version__}")
print(f"httpstan version: {httpstan.__version__}")

# Test model compilation
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
```

## Why This Process is Necessary

1. **PyPI Limitations**: The PyPI package only provides x86_64 wheels, which don't work natively on ARM64 Macs. While macOS can run x86_64 binaries under Rosetta emulation, this is slower and can cause timeout issues.

2. **stanc Compiler**: The `make` command downloads a pre-built x86_64 `stanc` binary. While this would work under Rosetta emulation, it's too slow and often exceeds httpstan's 1-second compilation timeout. We replace it with the native ARM64 version from Homebrew for better performance.

3. **Shared Libraries**: httpstan requires compiled C++ libraries (SUNDIALS, TBB) that must be built for your specific architecture. The `make` command builds these natively for ARM64.

**Important:** Your system runs natively in ARM64 mode. Rosetta emulation is only used when explicitly invoking x86_64 binaries (e.g., with `arch -x86_64`), but native ARM64 binaries are always preferred for performance.

## Troubleshooting

### Issue: stanc times out or fails

**Solution:** Make sure you've replaced the x86_64 `stanc` with the ARM64 version from Homebrew (Step 4.3).

### Issue: Import errors

**Solution:** Ensure you've activated your virtual environment and that both packages are installed in the same environment.

### Issue: Compilation errors during `make`

**Solution:** 
- Ensure Xcode Command Line Tools are installed: `xcode-select --install`
- Check that you have clang ≥10.0: `clang --version`

## Summary

The key steps for ARM64 installation are:
1. Install `stanc3` via Homebrew (ARM64 version)
2. Run `make` to build shared libraries
3. Replace the x86_64 `stanc` with the ARM64 version
4. Install httpstan from source
5. Install pystan

This process ensures all components are built for ARM64 architecture and compatible with Apple Silicon Macs.

