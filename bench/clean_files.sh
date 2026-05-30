#!/bin/bash
set -e

RESULT_DIR="./result"

if [ ! -d "$RESULT_DIR" ]; then
    echo "Error: $RESULT_DIR does not exist"
    exit 1
fi

echo "Finding all 'files' directories under $RESULT_DIR..."
FILES_DIRS=$(find "$RESULT_DIR" -type d -name "files")

if [ -z "$FILES_DIRS" ]; then
    echo "No 'files' directories found."
    exit 0
fi

echo "The following directories will be deleted:"
echo "$FILES_DIRS"
echo ""

TOTAL_SIZE=$(du -ch $FILES_DIRS 2>/dev/null | tail -1 | cut -f1)
echo "Total size to be freed: $TOTAL_SIZE"
echo ""

read -p "Are you sure you want to delete these directories? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "Deleting..."
    find "$RESULT_DIR" -type d -name "files" -exec rm -rf {} + 2>/dev/null || true
    echo "Done."
else
    echo "Aborted."
fi
