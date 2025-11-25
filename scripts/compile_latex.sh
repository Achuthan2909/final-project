#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LATEX_DIR="${PROJECT_DIR}/latex"
BUILD_DIR="${LATEX_DIR}"

cd "$PROJECT_DIR"
mkdir -p "$LATEX_DIR"

compile_tex() {
    local TEX_FILE="$1"
    local TEX_PATH
    
    if [[ "$TEX_FILE" == /* ]]; then
        TEX_PATH="$TEX_FILE"
    elif [[ "$TEX_FILE" == */* ]]; then
        TEX_PATH="$PROJECT_DIR/$TEX_FILE"
    else
        TEX_PATH="$LATEX_DIR/$TEX_FILE"
    fi
    
    if [ ! -f "$TEX_PATH" ]; then
        echo "Error: File '$TEX_PATH' not found"
        return 1
    fi
    
    local TEX_DIR="$(dirname "$TEX_PATH")"
    local TEX_NAME="$(basename "$TEX_PATH")"
    local PDF_NAME="${TEX_NAME%.tex}.pdf"
    
    echo "Compiling: $TEX_NAME"
    cd "$TEX_DIR"
    
    pdflatex -output-directory="$BUILD_DIR" "$TEX_NAME"
    pdflatex -output-directory="$BUILD_DIR" "$TEX_NAME"
    
    if [ -f "$BUILD_DIR/$PDF_NAME" ]; then
        cp "$BUILD_DIR/$PDF_NAME" "$LATEX_DIR/"
        echo "✓ PDF created: $LATEX_DIR/$PDF_NAME"
        echo "  Auxiliary files are in: $BUILD_DIR/"
    else
        echo "✗ Failed to create PDF for $TEX_NAME"
        return 1
    fi
}

if [ $# -eq 0 ]; then
    echo "Compiling all .tex files in latex/ directory..."
    cd "$LATEX_DIR"
    for tex_file in *.tex; do
        if [ -f "$tex_file" ]; then
            compile_tex "$tex_file"
            echo ""
        fi
    done
    if [ ! -f "$LATEX_DIR"/*.tex ]; then
        echo "No .tex files found in $LATEX_DIR/"
    fi
else
    for tex_file in "$@"; do
        compile_tex "$tex_file"
        echo ""
    done
fi

