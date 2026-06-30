trnazap infer \
--input "./data/yeast/yeast.pod5" \
--config "../configs/zap_s54_c79_BIOyeast_fragmented.yaml" \
--output "./data/yeast/yeast_zap_inference.zir" \
--batch-size 1024 \
--force-cpu

trnazap align \
--unaligned_bam "./data/yeast/yeast_unaligned_dorado_1.0.0.emit_moves.calmd.bam" \
--inference "./data/yeast/yeast_zap_inference.zir" \
--out_dir "./data/yeast" \
--out_pre "yeast_aligned_zap" \
--threads 4 \
--model yeast \
--wf_gap_open 1 \
--wf_gap_extend 0.7 \
--sw_gap_open -6 \
--sw_gap_extend -3 \
--sw_match 3 \
--sw_mismatch -3

trnazap alignment_visualize \
    --model yeast \
    --out_dir "./bwa_mem_vs_zap_comp_results/yeast/" \
    --out_prefix "yeast_" \
    --threads 8 \
    --bwa_path "./data/yeast/yeast_aligned_bwa.bam" \
    --zap_path "./data/yeast/yeast_aligned_zap.sorted.bam"