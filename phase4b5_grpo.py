"""
Phase 4b.5: GRPO Reasoning RL.
Run AFTER Phase 4b (CoT SFT) is complete.

What this phase does:
- Loads the CoT SFT checkpoint
- Runs GRPO for 700 steps on math and code only
- Uses rule-based rewards (no neural reward models)
- Saves final GRPO checkpoint for Phase 4c

What this phase does NOT do:
- Does NOT train on emotion/empathy/writing (no verifiable reward)
- Does NOT use GRPO with more than 1000 steps (reward hacking risk)
- Does NOT modify the tokenizer or model architecture

Dataset: Mix of GSM8K + MATH subset + OpenR1-Math subset
         Python-Edu code problems with test cases
"""
import torch, glob
from slm_project.config import ModelConfig, TrainConfig, GRPOConfig
from slm_project.model.model import SLM
from slm_project.training.grpo_trainer import GRPOTrainer
from slm_project.tokenizer_utils import load_tokenizer

def main():
    device = 'cuda'
    cfg    = ModelConfig()
    tcfg   = TrainConfig()
    gcfg   = GRPOConfig()
    tok    = load_tokenizer()

    # ── Load Phase 4b SFT checkpoint ─────────────────────────────
    sft_ckpts = sorted(glob.glob('trained_models/sft_cot_*.pt'))
    if not sft_ckpts:
        raise FileNotFoundError(
            "No Phase 4b CoT SFT checkpoint found.\n"
            "Run phase4b_cot_sft.py first."
        )
    latest_sft = sft_ckpts[-1]
    print(f"Loading Phase 4b checkpoint: {latest_sft}")

    ckpt  = torch.load(latest_sft, map_location='cpu')
    model = SLM(cfg, tcfg).to(device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Model loaded: {model.get_num_params():,} params")

    # ── Verify model has IHA in global layers ──────────────────
    from slm_project.model.attention import IHAGlobalAttention
    iha_count = sum(1 for n, m in model.named_modules()
                    if isinstance(m, IHAGlobalAttention))
    assert iha_count == 4, \
        f"Expected 4 IHAGlobalAttention modules, found {iha_count}. " \
        f"Rebuild model with Stage B changes."

    # ── Load GRPO math/code dataset ────────────────────────────
    def load_grpo_dataset():
        """
        Load verifiable math and code problems for GRPO.
        Each example must have:
          prompt_ids:   list of token IDs
          ground_truth: correct answer string
          task_type:    'math' or 'code'
          test_cases:   list of (input, expected_output) for code tasks
        """
        from datasets import load_dataset
        examples = []

        # GSM8K: grade school math with verifiable answers
        gsm8k = load_dataset('openai/gsm8k', 'main', split='train')
        for ex in gsm8k:
            # Extract numeric answer from GSM8K format: "#### 42"
            answer_match = ex['answer'].split('####')[-1].strip()

            # Build prompt in your chat format
            prompt = (
                f"<|user|>{ex['question']}<|end|>"
                f"<|assistant|><|think|>"  # Open think block — model must close it
            )
            prompt_ids = tok.encode(prompt, add_special_tokens=False)

            examples.append({
                'prompt_ids':   prompt_ids,
                'ground_truth': answer_match,
                'task_type':    'math',
                'test_cases':   None,
            })
            if len(examples) >= 5000:   # Cap at 5K math examples
                break

        print(f"Loaded {len(examples)} GSM8K examples for GRPO")

        # NOTE: Add code examples similarly using Python-Edu or HumanEval-style problems
        # with compilation test cases. Omitted here for brevity — add before running.

        return examples

    dataset = load_grpo_dataset()
    print(f"GRPO dataset: {len(dataset)} examples")

    # ── Run GRPO ──────────────────────────────────────────────────
    trainer = GRPOTrainer(model, gcfg, device=device)
    success = trainer.run(dataset)

    if success:
        # Save final GRPO checkpoint
        torch.save({
            'model_state': model.state_dict(),
            'grpo_steps':  trainer.step,
        }, 'trained_models/grpo_final.pt')
        print("\nPhase 4b.5 GRPO complete.")
        print("Checkpoint saved: trained_models/grpo_final.pt")
        print("Proceed to Phase 4c: Domain Fine-Tuning.")
    else:
        print("\nGRPO halted early — check KL divergence.")
        print("Load last GRPO checkpoint and proceed to Phase 4c.")

if __name__ == '__main__':
    main()
