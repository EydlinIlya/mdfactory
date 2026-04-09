#!/bin/bash
# optional:
# conda activate mdfactory

# SET VARIABLES
CSV_FILE="test.csv"
OUTPUT_DIR="testfolder"

# Derive summary YAML filename from CSV stem (basename only)
CSV_BASENAME="$(basename "${CSV_FILE}")"
SUMMARY_YAML="${CSV_BASENAME%.csv}.yaml"

# Step 1: Build systems from CSV
nextflow run build.nf --csv_file "${CSV_FILE}" --output_dir "${OUTPUT_DIR}"

# Step 2: Run GROMACS simulation chain
nextflow run simulate.nf -c simulate.config --base_dir "${OUTPUT_DIR}" --config_yaml "${OUTPUT_DIR}/${SUMMARY_YAML}"
