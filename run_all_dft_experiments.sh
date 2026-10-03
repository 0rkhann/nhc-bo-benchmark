#!/bin/bash
# Comprehensive script to run all DFT experiments
# Part 1: Bayesian Optimization experiments
# Part 2: Random Search baseline experiments
#
# All experiments use the xTB target energies in dft_G.json (named after the DFT descriptor set)
# Datasets: dft_descriptors, dft_chemberta2, dft_mordred (3 DFT datasets)
# Methods: fabo, pls, pca, opls (4 methods)
# Kernel: Matern
# Acquisition: EI
# Seeds: 42, 43, 44, 45, 46 (5 seeds)
# Total BO: 3 × 4 × 5 = 60 experiments
# Total Random: 3 × 5 = 15 experiments
# Grand Total: 75 experiments

# Create logs directory
mkdir -p logs

# Activate virtual environment if it exists
if [ -d ".bo_project_env" ]; then
    source .bo_project_env/bin/activate
fi

# ============================================================================
# PART 1: BAYESIAN OPTIMIZATION EXPERIMENTS
# ============================================================================

echo "========================================================================"
echo "PART 1: BAYESIAN OPTIMIZATION EXPERIMENTS"
echo "========================================================================"

# Define BO experiment configurations
BO_DATASETS=("dft_descriptors.csv:dft_G.json" "dft_chemberta2.csv:dft_G.json" "dft_mordred.csv:dft_G.json")
METHODS=("fabo" "pls" "pca" "opls")
KERNEL="Matern"
ACQUISITION="EI"
SEEDS=(42 43 44 45 46)

# Calculate total BO experiments
BO_TOTAL=$((${#BO_DATASETS[@]} * ${#METHODS[@]} * ${#SEEDS[@]}))
echo "Total BO experiments to run: $BO_TOTAL"
echo "Datasets: ${#BO_DATASETS[@]} DFT datasets"
echo "Kernel: $KERNEL"
echo "Acquisition: $ACQUISITION"
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "========================================================================"
echo ""

BO_COUNTER=0
BO_FAILED=0

# Loop over all BO combinations
for DATASET_ENTRY in "${BO_DATASETS[@]}"; do
    IFS=':' read -r DATASET CACHE <<< "$DATASET_ENTRY"
    DATASET_NAME=$(basename "$DATASET" .csv)
    
    for METHOD in "${METHODS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            BO_COUNTER=$((BO_COUNTER + 1))
            OUTPUT_DIR="results/${DATASET_NAME}/${METHOD}_${KERNEL}_${ACQUISITION}/seed${SEED}"
            
            echo "========================================================================"
            echo "BO Experiment [$BO_COUNTER/$BO_TOTAL]"
            echo "  Dataset: $DATASET_NAME"
            echo "  Method: $METHOD"
            echo "  Kernel: $KERNEL"
            echo "  Acquisition: $ACQUISITION"
            echo "  Seed: $SEED"
            echo "  Output: $OUTPUT_DIR"
            echo "========================================================================"
            
            python -m src.cli \
                --mode "$METHOD" \
                --input "data/$DATASET" \
                --cache "data/$CACHE" \
                --output-dir "$OUTPUT_DIR" \
                --n-initial 10 \
                --n-iter 100 \
                --seed "$SEED" \
                --repeats 1 \
                --kernels "$KERNEL" \
                --acquisitions "$ACQUISITION" \
                --betas "1.0" \
                2>&1 | tee "logs/${DATASET_NAME}_${METHOD}_${KERNEL}_${ACQUISITION}_seed${SEED}.log"
            
            EXIT_CODE=${PIPESTATUS[0]}
            if [ $EXIT_CODE -ne 0 ]; then
                echo "❌ ERROR: BO Experiment failed with exit code $EXIT_CODE"
                BO_FAILED=$((BO_FAILED + 1))
                echo "Continuing with next experiment..."
            else
                echo "✅ SUCCESS: BO Experiment completed"
            fi
            echo ""
        done
    done
done

echo "========================================================================"
echo "PART 1 COMPLETE: Bayesian Optimization"
echo "  Total run: $BO_COUNTER experiments"
echo "  Successful: $((BO_COUNTER - BO_FAILED))"
echo "  Failed: $BO_FAILED"
echo "========================================================================"
echo ""
echo ""

# ============================================================================
# PART 2: RANDOM SEARCH BASELINE EXPERIMENTS
# ============================================================================

echo "========================================================================"
echo "PART 2: RANDOM SEARCH BASELINE EXPERIMENTS"
echo "========================================================================"

# Define Random Search experiment configurations (same 3 DFT datasets)
RANDOM_DATASETS=("dft_descriptors.csv:dft_G.json" "dft_chemberta2.csv:dft_G.json" "dft_mordred.csv:dft_G.json")

# Calculate total random experiments
RANDOM_TOTAL=$((${#RANDOM_DATASETS[@]} * ${#SEEDS[@]}))
echo "Total Random Search experiments to run: $RANDOM_TOTAL"
echo "Datasets: ${#RANDOM_DATASETS[@]} DFT datasets"
echo "Seeds: ${SEEDS[*]}"
echo "========================================================================"
echo ""

RANDOM_COUNTER=0
RANDOM_FAILED=0

# Loop over all random search combinations
for DATASET_ENTRY in "${RANDOM_DATASETS[@]}"; do
    IFS=':' read -r DATASET CACHE <<< "$DATASET_ENTRY"
    DATASET_NAME=$(basename "$DATASET" .csv)
    
    for SEED in "${SEEDS[@]}"; do
        RANDOM_COUNTER=$((RANDOM_COUNTER + 1))
        OUTPUT_DIR="results/${DATASET_NAME}_random_search/seed${SEED}"
        
        echo "========================================================================"
        echo "Random Experiment [$RANDOM_COUNTER/$RANDOM_TOTAL]"
        echo "  Dataset: $DATASET_NAME"
        echo "  Method: random"
        echo "  Seed: $SEED"
        echo "  Output: $OUTPUT_DIR"
        echo "========================================================================"
        
        python -m src.cli \
            --mode "random" \
            --input "data/$DATASET" \
            --cache "data/$CACHE" \
            --output-dir "results" \
            --n-initial 10 \
            --n-iter 100 \
            --seed "$SEED" \
            --repeats 1 \
            2>&1 | tee "logs/${DATASET_NAME}_random_seed${SEED}.log"
        
        EXIT_CODE=${PIPESTATUS[0]}
        if [ $EXIT_CODE -ne 0 ]; then
            echo "❌ ERROR: Random Experiment failed with exit code $EXIT_CODE"
            RANDOM_FAILED=$((RANDOM_FAILED + 1))
            echo "Continuing with next experiment..."
        else
            echo "✅ SUCCESS: Random Experiment completed"
        fi
        echo ""
    done
done

echo "========================================================================"
echo "PART 2 COMPLETE: Random Search Baselines"
echo "  Total run: $RANDOM_COUNTER experiments"
echo "  Successful: $((RANDOM_COUNTER - RANDOM_FAILED))"
echo "  Failed: $RANDOM_FAILED"
echo "========================================================================"
echo ""
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo "Bayesian Optimization:"
echo "  Total: $BO_COUNTER"
echo "  Successful: $((BO_COUNTER - BO_FAILED))"
echo "  Failed: $BO_FAILED"
echo ""
echo "Random Search:"
echo "  Total: $RANDOM_COUNTER"
echo "  Successful: $((RANDOM_COUNTER - RANDOM_FAILED))"
echo "  Failed: $RANDOM_FAILED"
echo ""
echo "Grand Total: $((BO_COUNTER + RANDOM_COUNTER)) experiments"
echo "Grand Success: $((BO_COUNTER - BO_FAILED + RANDOM_COUNTER - RANDOM_FAILED))"
echo "Grand Failed: $((BO_FAILED + RANDOM_FAILED))"
echo "========================================================================"

