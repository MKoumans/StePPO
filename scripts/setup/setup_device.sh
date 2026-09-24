#!/bin/bash
set -e
#
# Checks Docker, Docker Compose, and NVIDIA GPU/Container Toolkit
# availability, then builds the Docker image. Interactive: prompts for
# confirmation if GPU support isn't fully working. Takes no arguments.
#
# Usage:
#   ./scripts/setup/setup_device.sh

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $(basename "$0")"
    echo ""
    echo "Checks Docker, Docker Compose, and NVIDIA GPU/Container Toolkit"
    echo "availability, then builds the Docker image. Interactive: prompts for"
    echo "confirmation if GPU support isn't fully working. Takes no arguments."
    exit 0
fi

echo "========================================="
echo "   VariBAD JAX - Device Setup Script     "
echo "========================================="

# 1. Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[-] Error: Docker is not installed. Please install Docker."
    echo "    Visit: https://docs.docker.com/engine/install/"
    exit 1
fi
echo "[+] Docker is installed."

# 2. Check if Docker Compose is installed
if ! docker compose version &> /dev/null; then
    echo "[-] Error: 'docker compose' (Compose V2) is not installed."
    echo "    Please install docker-compose-plugin."
    exit 1
fi
echo "[+] Docker Compose is installed."

# 3. Check for NVIDIA Driver & Docker GPU Support
if command -v nvidia-smi &> /dev/null; then
    echo "[+] NVIDIA GPU detected:"
    nvidia-smi -L

    echo "[*] Checking NVIDIA Container Toolkit integration with Docker..."
    if docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "[+] GPU integration is working! JAX inside Docker will be able to use your GPU."
    else
        echo "[!] WARNING: NVIDIA Container Toolkit is not configured for Docker."
        echo "    Without it, Docker containers cannot access the GPU and JAX will fall back to CPU."
        echo "    Install it via: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
        echo ""
        read -p "Do you want to continue building the image anyway? [y/N]: " -r response
        if [[ ! "$response" =~ ^[yY]$ ]]; then
            echo "Aborting setup."
            exit 1
        fi
    fi
else
    echo "[!] WARNING: No NVIDIA GPU detected (nvidia-smi not found)."
    echo "    If you run on this device, training will run on CPU only (which will be slow)."
    read -p "Do you want to continue? [y/N]: " -r response
    if [[ ! "$response" =~ ^[yY]$ ]]; then
        echo "Aborting setup."
        exit 1
    fi
fi

# 4. Build Docker container
echo "[*] Building the Docker image (this might take a few minutes as it installs JAX and dependencies)..."
docker compose build

echo ""
echo "[+] Setup completed successfully!"
echo "    You can now run training using the helper scripts:"
echo "    - GridWorld:  ./scripts/run_gridworld.sh"
echo "    - ODE Solver: ./scripts/run_ode.sh"
echo "========================================="
