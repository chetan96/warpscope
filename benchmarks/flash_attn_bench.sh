#!/usr/bin/env bash
# flash_attn_bench.sh — Full Flash Attention 2 benchmark + warpscope analysis pipeline.
#
# Builds baseline and specialized FA2 kernels, runs them with and without
# warpscope, and produces a comparison table showing TFLOPS, latency, and
# idle warp patterns (causal_triangle, balanced, etc.).
#
# Usage: ./flash_attn_bench.sh [--shapes FILTER] [--skip-build]
#
# Environment:
#   BENCH_ARCH   GPU arch (default: sm_89)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$PROJECT_ROOT/examples"
ARCH="${BENCH_ARCH:-sm_89}"

# Parse args
FILTER=""
SKIP_BUILD=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --shapes) FILTER="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Find warpscope.so
WARPSCOPE_SO=""
for candidate in \
    "$PROJECT_ROOT/warpscope.so" \
    "$PROJECT_ROOT/nvbit_release/tools/warpscope/warpscope.so"; do
    if [[ -f "$candidate" ]]; then
        WARPSCOPE_SO="$(realpath "$candidate")"
        break
    fi
done
if [[ -z "$WARPSCOPE_SO" ]]; then
    echo "ERROR: warpscope.so not found. Run 'make' in the project root first."
    exit 1
fi

# Results directory
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/results/flash_attn_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "=== Flash Attention 2 Benchmark Pipeline ==="
echo "  Arch:       $ARCH"
echo "  Warpscope:  $WARPSCOPE_SO"
echo "  Results:    $RESULTS_DIR"
echo "  Filter:     ${FILTER:-<all>}"
echo ""

# ─── Step 1: Build ───────────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false ]]; then
    echo ">>> Step 1: Building kernels..."
    cd "$EXAMPLES_DIR"
    nvcc -arch=$ARCH -lineinfo flash_attention_bench.cu -o flash_attention_bench 2>&1 | tee "$RESULTS_DIR/build_baseline.log"
    nvcc -arch=$ARCH -lineinfo flash_attention_specialized.cu -o flash_attention_specialized 2>&1 | tee "$RESULTS_DIR/build_specialized.log"
    echo "  Build complete."
else
    echo ">>> Step 1: Skipping build (--skip-build)"
fi
echo ""

# ─── Step 2: Baseline timing (no warpscope) ─────────────────────────────────
echo ">>> Step 2: Baseline timing..."
cd "$EXAMPLES_DIR"

BENCH_ARGS=""
[[ -n "$FILTER" ]] && BENCH_ARGS="$FILTER"

./flash_attention_bench $BENCH_ARGS 2>&1 | tee "$RESULTS_DIR/baseline_timing.txt"
echo ""

# ─── Step 3: Warpscope analysis of baseline ──────────────────────────────────
echo ">>> Step 3: Warpscope analysis of baseline..."
WARPSCOPE_JSON="$RESULTS_DIR/warpscope_baseline.json" \
    LD_PRELOAD="$WARPSCOPE_SO" \
    ./flash_attention_bench $BENCH_ARGS 2>"$RESULTS_DIR/warpscope_baseline.log" | tee "$RESULTS_DIR/baseline_with_warpscope.txt"
echo "  Warpscope baseline log: $RESULTS_DIR/warpscope_baseline.log"
echo "  Warpscope baseline JSON: $RESULTS_DIR/warpscope_baseline.json"
echo ""

# ─── Step 4: Specialized kernel comparison ───────────────────────────────────
echo ">>> Step 4: Specialized kernel timing..."
./flash_attention_specialized $BENCH_ARGS 2>&1 | tee "$RESULTS_DIR/specialized_timing.txt"
echo ""

# ─── Step 5: Warpscope analysis of specialized ──────────────────────────────
echo ">>> Step 5: Warpscope analysis of specialized kernels..."
WARPSCOPE_JSON="$RESULTS_DIR/warpscope_specialized.json" \
    LD_PRELOAD="$WARPSCOPE_SO" \
    ./flash_attention_specialized $BENCH_ARGS 2>"$RESULTS_DIR/warpscope_specialized.log" | tee "$RESULTS_DIR/specialized_with_warpscope.txt"
echo "  Warpscope specialized log: $RESULTS_DIR/warpscope_specialized.log"
echo "  Warpscope specialized JSON: $RESULTS_DIR/warpscope_specialized.json"
echo ""

# ─── Step 6: Comparison report ───────────────────────────────────────────────
echo ">>> Step 6: Generating comparison report..."

REPORT="$RESULTS_DIR/comparison.txt"

cat > "$REPORT" <<'HEADER'
======================================================================
  Flash Attention 2: Baseline vs Warp-Specialized Comparison
======================================================================

HEADER

# Extract RESULT lines from baseline
echo "--- Baseline Performance ---" >> "$REPORT"
printf "%-22s %6s %4s %5s %7s %6s %6s %8s %8s\n" \
    "Name" "SeqLen" "Dim" "BlkM" "Causal" "Blocks" "Warps" "ms" "TFLOPS" >> "$REPORT"
printf "%-22s %6s %4s %5s %7s %6s %6s %8s %8s\n" \
    "─────" "──────" "───" "────" "──────" "──────" "─────" "──" "──────" >> "$REPORT"
grep '^RESULT' "$RESULTS_DIR/baseline_timing.txt" | while IFS=$'\t' read -r _ name seq dim bm causal nblocks wpb tw ms tflops; do
    printf "%-22s %6s %4s %5s %7s %6s %6s %8s %8s\n" \
        "$name" "$seq" "$dim" "$bm" "$causal" "$nblocks" "$tw" "$ms" "$tflops" >> "$REPORT"
done
echo "" >> "$REPORT"

# Extract COMPARE lines from specialized
echo "--- Specialized vs Baseline ---" >> "$REPORT"
printf "%-22s %8s %8s %8s %8s %6s %8s %8s %6s %8s %8s %6s\n" \
    "Name" "Base ms" "Base TF" "DBuf ms" "DBuf TF" "DBuf x" "Dyn ms" "Dyn TF" "Dyn x" "Repr ms" "Repr TF" "Repr x" >> "$REPORT"
printf "%-22s %8s %8s %8s %8s %6s %8s %8s %6s %8s %8s %6s\n" \
    "─────" "───────" "───────" "───────" "───────" "─────" "──────" "──────" "────" "───────" "───────" "─────" >> "$REPORT"
grep '^COMPARE' "$RESULTS_DIR/specialized_timing.txt" | while IFS=$'\t' read -r _ name bms btf dms dtf dsx dyms dytf dysx rms rtf rsx; do
    if [[ "$dms" == "-" ]]; then
        printf "%-22s %8s %8s %8s %8s %6s %8s %8s %6s %8s %8s %6s\n" \
            "$name" "$bms" "$btf" "-" "-" "-" "-" "-" "-" "-" "-" "-" >> "$REPORT"
    else
        printf "%-22s %8s %8s %8s %8s %6s %8s %8s %6s %8s %8s %6s\n" \
            "$name" "$bms" "$btf" "$dms" "$dtf" "${dsx}x" "$dyms" "$dytf" "${dysx}x" "$rms" "$rtf" "${rsx}x" >> "$REPORT"
    fi
done
echo "" >> "$REPORT"

# Extract idle warp info from warpscope logs
echo "--- Warpscope Idle Warp Summary ---" >> "$REPORT"
echo "Baseline:" >> "$REPORT"
grep -E "Idle|idle|Total warps|pattern|causal" "$RESULTS_DIR/warpscope_baseline.log" 2>/dev/null | head -40 >> "$REPORT" || echo "  (no idle warp data)" >> "$REPORT"
echo "" >> "$REPORT"
echo "Specialized:" >> "$REPORT"
grep -E "Idle|idle|Total warps|pattern|causal" "$RESULTS_DIR/warpscope_specialized.log" 2>/dev/null | head -40 >> "$REPORT" || echo "  (no idle warp data)" >> "$REPORT"
echo "" >> "$REPORT"

echo "======================================================================" >> "$REPORT"
echo "Report saved to: $REPORT" >> "$REPORT"

cat "$REPORT"

echo ""
echo "=== Pipeline complete ==="
echo "All results in: $RESULTS_DIR"
echo ""
echo "Files:"
ls -la "$RESULTS_DIR/"
