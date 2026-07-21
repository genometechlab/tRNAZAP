#!/bin/bash

# User-configurable paths 
SAMPLE="yeast"
DATA_DIR="./data/${SAMPLE}"
CONFIG="../configs/zap_s54_c79_BIOyeast_fragmented.yaml"
POD5="${DATA_DIR}/${SAMPLE}.pod5"
UNALIGNED_BAM="${DATA_DIR}/${SAMPLE}_unaligned_dorado_1.0.0.emit_moves.calmd.bam"
BWA_BAM="${DATA_DIR}/${SAMPLE}_aligned_bwa.bam"
COMP_DIR="./bwa_mem_vs_zap_comp_results/${SAMPLE}"
MODEL="yeast"
BATCH_SIZE=1024
THREADS=4

echo "=== Starting tRNAZAP pipeline for sample: $SAMPLE ==="
echo "Threads: $THREADS"

echo "=== Step 1: Zap Inference ==="
trnazap infer \
    --input "$POD5" \
    --config "$CONFIG" \
    --output "${DATA_DIR}/${SAMPLE}_zap_inference.zir" \
    --batch-size "$BATCH_SIZE" \
    --force-cpu
echo "Zap Inference complete."

echo "=== Step 2: Zap Alignment ==="
trnazap align \
    --unaligned_bam "$UNALIGNED_BAM" \
    --inference "${DATA_DIR}/${SAMPLE}_zap_inference.zir" \
    --out_dir "$DATA_DIR" \
    --out_pre "${SAMPLE}_aligned_zap" \
    --threads "$THREADS" \
    --model "$MODEL" \
    --wf_gap_open 1 \
    --wf_gap_extend 0.7 \
    --sw_gap_open -6 \
    --sw_gap_extend -3 \
    --sw_match 3 \
    --sw_mismatch -3
echo "Zap Alignment complete."

echo "=== Step 3: Alignment Visualization ==="
trnazap alignment_visualize \
    --model "$MODEL" \
    --out_dir "${COMP_DIR}/" \
    --out_prefix "${SAMPLE}_" \
    --threads "$THREADS" \
    --bwa_path "$BWA_BAM" \
    --zap_path "${DATA_DIR}/${SAMPLE}_aligned_zap.sorted.bam"
echo "Visualization complete."

echo "=== tRNAZAP pipeline finished ==="