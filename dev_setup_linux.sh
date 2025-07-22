#!/bin/bash

# =============================================================================
# Pygame Development Setup Script for Linux
# =============================================================================
# 
# This script automates the setup of a development environment for pygame
# on Linux systems. It installs all necessary system dependencies, creates
# a Python virtual environment, and installs all required Python packages.
#
# This script is IDEMPOTENT - safe to run multiple times. It will:
# - Skip system packages that are already installed
# - Reuse existing virtual environment if it's working
# - Ask before rebuilding pygame if it's already installed
#
# Usage:
#   chmod +x dev_setup_linux.sh
#   ./dev_setup_linux.sh
#
# Requirements:
#   - Ubuntu/Debian-based Linux system (tested on Ubuntu 24.04)
#   - sudo privileges for installing system packages
#   - Python 3.6+ installed
#
# What this script does:
#   1. Updates system package manager
#   2. Installs SDL2 and other system dependencies
#   3. Creates a Python virtual environment (pygame_venv)
#   4. Installs Python development dependencies (no requirements.txt needed)
#   5. Runs pygame build configuration
#   6. Builds and installs pygame from source
#   7. Verifies the installation
#
# Author: Generated for pygame development setup
# Date: $(date +%Y-%m-%d)
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

# Check if running on a supported system
check_system() {
    print_header "Checking System Compatibility"
    
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_error "This script is designed for Linux systems only."
        exit 1
    fi
    
    # Check if we have apt (Debian/Ubuntu)
    if ! command -v apt &> /dev/null; then
        print_error "This script requires apt package manager (Ubuntu/Debian)."
        print_error "For other Linux distributions, please install dependencies manually."
        exit 1
    fi
    
    # Check if Python 3 is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed."
        print_error "Please install Python 3 first: sudo apt install python3"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "System check passed - Python $PYTHON_VERSION detected"
}

# Install system dependencies
install_system_dependencies() {
    print_header "Installing System Dependencies"
    
    # Check if we need to install packages
    PACKAGES="libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev libportmidi-dev libjpeg-dev libpng-dev python3-dev python3-venv python3-pip build-essential pkg-config"
    MISSING_PACKAGES=""
    
    print_status "Checking for missing system packages..."
    for pkg in $PACKAGES; do
        if ! dpkg -l | grep -q "^ii  $pkg "; then
            MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
        fi
    done
    
    if [ -n "$MISSING_PACKAGES" ]; then
        print_status "Updating package manager..."
        sudo apt update
        
        print_status "Installing missing packages:$MISSING_PACKAGES"
        sudo apt install -y $MISSING_PACKAGES
        print_success "System dependencies installed successfully"
    else
        print_success "All system dependencies are already installed"
    fi
}

# Create and setup Python virtual environment
setup_python_environment() {
    print_header "Setting Up Python Virtual Environment"
    
    # Check if venv already exists and is working
    if [ -d "pygame_venv" ] && [ -f "pygame_venv/bin/activate" ]; then
        print_status "Virtual environment already exists. Testing..."
        if pygame_venv/bin/python -c "import sys; print(sys.version)" &> /dev/null; then
            print_warning "Virtual environment exists and is functional. Keeping existing environment."
            print_status "Activating existing virtual environment..."
            source pygame_venv/bin/activate
            
            print_status "Upgrading pip..."
            pip install --upgrade pip
            
            print_success "Existing Python virtual environment activated"
            return 0
        else
            print_warning "Existing virtual environment appears corrupted. Recreating..."
            rm -rf pygame_venv
        fi
    fi
    
    print_status "Creating Python virtual environment..."
    python3 -m venv pygame_venv
    
    print_status "Activating virtual environment..."
    source pygame_venv/bin/activate
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_success "Python virtual environment created and activated"
}

# Install Python dependencies
install_python_dependencies() {
    print_header "Installing Python Dependencies"
    
    print_status "Installing/upgrading build dependencies..."
    pip install --upgrade numpy setuptools wheel build cython
    
    print_status "Installing/upgrading development dependencies..."
    pip install --upgrade pytest pytest-benchmark pillow
    
    print_success "Python dependencies installed successfully"
}

# Configure and build pygame
build_pygame() {
    print_header "Configuring and Building Pygame"
    
    # Check if pygame is already installed and working
    if python -c "import pygame; print('pygame version:', pygame.version.ver); pygame.quit()" &> /dev/null; then
        print_warning "Pygame is already installed and working."
        read -p "Do you want to rebuild pygame anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_success "Skipping pygame build - using existing installation"
            return 0
        fi
    fi
    
    print_status "Running pygame build configuration..."
    python buildconfig/config.py
    
    if [ $? -eq 0 ]; then
        print_success "Build configuration completed successfully"
    else
        print_warning "Build configuration completed with warnings (missing optional dependencies)"
    fi
    
    print_status "Building and installing pygame from source..."
    pip install . --force-reinstall --no-deps
    
    print_success "Pygame built and installed successfully"
}

# Verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    print_status "Testing pygame import..."
    python -c "import pygame; print('pygame version:', pygame.version.ver); pygame.quit()"
    
    if [ $? -eq 0 ]; then
        print_success "Pygame installation verified successfully!"
    else
        print_error "Pygame installation verification failed"
        return 1
    fi
    
    print_status "Checking available pygame modules..."
    python -c "import pygame; print('Available modules:'); [print('  -', module) for module in sorted(dir(pygame)) if not module.startswith('_')]"
}

# Display final instructions
show_final_instructions() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}Pygame development environment has been set up successfully!${NC}\n"
    
    echo -e "${YELLOW}To use the environment:${NC}"
    echo -e "  1. Activate the virtual environment:"
    echo -e "     ${BLUE}source pygame_venv/bin/activate${NC}"
    echo -e ""
    echo -e "  2. Test pygame with an example:"
    echo -e "     ${BLUE}python -m pygame.examples.aliens${NC}"
    echo -e ""
    echo -e "  3. Run pygame tests:"
    echo -e "     ${BLUE}python -m pygame.tests${NC}"
    echo -e ""
    echo -e "  4. To deactivate the environment:"
    echo -e "     ${BLUE}deactivate${NC}"
    echo -e ""
    
    echo -e "${YELLOW}Development workflow:${NC}"
    echo -e "  - After making changes to pygame source code, reinstall with:"
    echo -e "    ${BLUE}pip install .${NC}"
    echo -e "  - Or for development mode (editable install):"
    echo -e "    ${BLUE}pip install -e .${NC}"
    echo -e ""
    
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "  - If you encounter build issues, try running:"
    echo -e "    ${BLUE}python buildconfig/config.py${NC}"
    echo -e "  - Check system dependencies are installed correctly"
    echo -e "  - Ensure virtual environment is activated before building"
    echo -e ""
    
    echo -e "${GREEN}Happy pygame development! 🎮${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "=================================================================="
    echo "           Pygame Development Setup Script for Linux"
    echo "=================================================================="
    echo -e "${NC}"
    echo "This script will set up a complete pygame development environment."
    echo "It will install system dependencies, create a virtual environment,"
    echo "and build pygame from source."
    echo ""
    
    # Ask for confirmation
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Setup cancelled by user"
        exit 0
    fi
    
    # Check if we're in the pygame source directory
    if [ ! -f "setup.py" ] || [ ! -d "src_c" ] || [ ! -d "buildconfig" ]; then
        print_error "This script must be run from the pygame source directory"
        print_error "Make sure you're in the directory containing setup.py"
        exit 1
    fi
    
    # Execute setup steps
    check_system
    install_system_dependencies
    setup_python_environment
    install_python_dependencies
    build_pygame
    verify_installation
    show_final_instructions
}

# Handle script interruption
trap 'print_error "Setup interrupted by user"; exit 130' INT

# Run main function
main "$@"
