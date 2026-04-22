pip install lm_eval

# Run all benchmarks on final checkpoint
python -m lm_eval \
    --model hf \
    --model_args pretrained=slm_project/,dtype=bfloat16 \
    --tasks hellaswag,arc_easy,arc_challenge,mmlu,winogrande,piqa \
    --num_fewshot 0 \
    --batch_size 4 \
    --output_path eval_results/standard_benchmarks.json

# GSM8K with CoT
python -m lm_eval \
    --model hf \
    --model_args pretrained=slm_project/,dtype=bfloat16 \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 4 \
    --output_path eval_results/gsm8k_cot.json

echo "Evaluation complete. Check eval_results/ for scores."
