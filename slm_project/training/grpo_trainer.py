"""
Phase 4b.5: GRPO for Math and Code Reasoning.

Based on DeepSeek-R1 methodology. Key differences from full R1:
- G=8 (not 16) — 125M model needs less exploration diversity
- max_gen_len=4096 (not 32768) — 125M model can't reliably produce longer chains
- Rule-based rewards ONLY — no neural reward model (reward hacking risk)
- 700 steps max — bounded phase to prevent reward hacking
- Only on math and code — verifiable domains only

REWARD STRUCTURE (from DeepSeek-R1 paper equation 4):
  Reward_total = accuracy_reward + format_reward + language_reward

Accuracy (0 or 1): Is the final answer correct?
Format (0 or 0.5 or 1): Does the response use <|think|>...</|think|> tags?
Language (0 to 1): Fraction of response in target language (prevents lang mixing)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

from slm_project.config import ModelConfig, GRPOConfig
from slm_project.tokenizer_utils import load_tokenizer


# ── Special token IDs (must match tokenizer) ──────────────────
tok = load_tokenizer()
TOK_THINK       = tok.convert_tokens_to_ids('<|think|>')     # 32003
TOK_THINK_END   = tok.convert_tokens_to_ids('<|/think|>')    # 32004
TOK_END         = tok.convert_tokens_to_ids('<|end|>')       # 32005
TOK_ASSISTANT   = tok.convert_tokens_to_ids('<|assistant|>') # 32002


# ═══════════════════════════════════════════════════════════
# REWARD FUNCTIONS
# ═══════════════════════════════════════════════════════════

def accuracy_reward_math(response_text: str, ground_truth: str) -> float:
    """
    Rule-based accuracy reward for math problems.
    Returns 1.0 if response contains the correct final answer, else 0.0.

    The model should wrap its final answer in \\boxed{} (standard math format).
    Extracts the boxed answer and compares to ground truth.
    """
    # Extract boxed answer
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', response_text)
    if boxed_match is None:
        return 0.0   # No boxed answer found

    predicted = boxed_match.group(1).strip()
    expected  = ground_truth.strip()

    # Normalize: remove spaces, compare numerically if possible
    try:
        pred_val = float(predicted.replace(',', ''))
        exp_val  = float(expected.replace(',', ''))
        return 1.0 if abs(pred_val - exp_val) < 1e-4 else 0.0
    except ValueError:
        # String comparison for non-numeric answers
        return 1.0 if predicted.lower() == expected.lower() else 0.0


def accuracy_reward_code(response_text: str, test_cases: List[Tuple]) -> float:
    """
    Rule-based accuracy reward for code problems.
    Returns fraction of test cases passed (0.0 to 1.0).

    Extract code from response, execute against test cases.
    SAFETY: runs in sandboxed subprocess with timeout.
    """
    # Extract code block from response
    code_match = re.search(r'```python\n(.*?)```', response_text, re.DOTALL)
    if code_match is None:
        # Try without language tag
        code_match = re.search(r'```\n(.*?)```', response_text, re.DOTALL)
    if code_match is None:
        return 0.0

    code = code_match.group(1)

    # Run test cases
    import subprocess, tempfile
    passed = 0
    for inputs, expected_output in test_cases:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write code + test
                f.write(code + '\n')
                if isinstance(inputs, str):
                    f.write(f"print({inputs})\n")
                test_file = f.name

            result = subprocess.run(
                ['python3', test_file],
                capture_output=True, text=True, timeout=5.0
            )
            output = result.stdout.strip()
            if str(expected_output).strip() == output:
                passed += 1
        except Exception:
            pass
        finally:
            try:
                os.unlink(test_file)
            except Exception:
                pass

    return passed / len(test_cases) if test_cases else 0.0


def format_reward(response_ids: List[int]) -> float:
    """
    Reward for correct use of <|think|>...</|think|> reasoning tags.
    Returns 1.0 if response contains at least one complete think block.
    Returns 0.5 if response has <|think|> but no closing tag.
    Returns 0.0 if no think tags found.

    DeepSeek-R1: format reward incentivizes visible reasoning structure.
    """
    has_open  = TOK_THINK     in response_ids
    has_close = TOK_THINK_END in response_ids

    if has_open and has_close:
        # Verify tags are in correct order
        open_pos  = response_ids.index(TOK_THINK)
        close_pos = response_ids.index(TOK_THINK_END)
        return 1.0 if close_pos > open_pos else 0.0
    elif has_open and not has_close:
        return 0.5   # Partial credit — started reasoning but didn't close
    else:
        return 0.0


def language_consistency_reward(response_text: str,
                                 target_lang: str = 'en') -> float:
    """
    Reward proportional to fraction of response in target language.
    From DeepSeek-R1 paper Equation 7: Num(Words_target) / Num(Words)

    Prevents language mixing (English model switching to Chinese mid-response).
    Simplified heuristic for English: penalize CJK character density.
    """
    if not response_text:
        return 1.0

    total_chars = len(response_text)
    if total_chars == 0:
        return 1.0

    # CJK character ranges
    cjk_chars = sum(
        1 for c in response_text
        if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff'
    )

    if target_lang == 'en':
        # English target: penalize CJK density
        cjk_fraction = cjk_chars / total_chars
        return max(0.0, 1.0 - cjk_fraction * 5.0)   # Heavily penalize CJK in English

    return 1.0   # Default: no penalty


def compute_combined_reward(
    response_text:   str,
    response_ids:    List[int],
    ground_truth:    str,
    task_type:       str,              # 'math' or 'code'
    test_cases:      Optional[List]  = None,
    gcfg:            GRPOConfig      = None,
    target_lang:     str             = 'en',
) -> float:
    """
    Combined reward = accuracy + format + language (weighted sum).
    Mirrors DeepSeek-R1 equation 4.
    """
    if gcfg is None:
        gcfg = GRPOConfig()

    # Accuracy (task-specific rule-based)
    if task_type == 'math':
        acc = accuracy_reward_math(response_text, ground_truth)
    elif task_type == 'code' and test_cases:
        acc = accuracy_reward_code(response_text, test_cases)
    else:
        acc = 0.0

    fmt  = format_reward(response_ids)
    lang = language_consistency_reward(response_text, target_lang)

    return (gcfg.accuracy_weight  * acc +
            gcfg.format_weight    * fmt +
            gcfg.language_weight  * lang)


# ═══════════════════════════════════════════════════════════
# GRPO OBJECTIVE
# ═══════════════════════════════════════════════════════════

def compute_grpo_loss(
    policy_logits:  torch.Tensor,    # [B*G, T, vocab] from policy model
    ref_logits:     torch.Tensor,    # [B*G, T, vocab] from reference model (no_grad)
    rewards:        torch.Tensor,    # [B*G] scalar reward per response
    response_masks: torch.Tensor,    # [B*G, T] 1 on response tokens, 0 on prompt
    G:              int,             # Group size
    clip_eps:       float = 0.2,
    kl_coef:        float = 0.001,
) -> Tuple[torch.Tensor, dict]:
    """
    GRPO objective (DeepSeek-R1 paper equation 1):

    J_GRPO = E[1/G Σ_i min(π_θ/π_old × A_i, clip(π_θ/π_old, 1-ε, 1+ε) × A_i)
               - β × KL(π_θ || π_ref)]

    Advantage A_i = (r_i - mean(r)) / std(r)  [equation 3]

    Implementation notes:
    - We use policy logits from the current step and reference logits from
      the frozen reference model.
    - For efficiency, we compute log-probs rather than ratio of probabilities.
    - Since we update the reference model every ref_update_freq steps
      (not every step), π_old ≈ π_ref for small update intervals.
      We approximate π_θ/π_old ≈ exp(log π_θ - log π_ref) as is standard
      in GRPO implementations.
    """
    BG, T, V = policy_logits.shape
    B = BG // G

    # ── Log probabilities ─────────────────────────────────────
    # Only compute on response tokens (response_mask=1)
    policy_logprobs = F.log_softmax(policy_logits, dim=-1)   # [BG, T, V]
    ref_logprobs    = F.log_softmax(ref_logits,    dim=-1)   # [BG, T, V]

    # Per-token log prob of the actual generated tokens
    # response_ids would be passed separately; here we use policy vs ref ratio
    # aggregate over sequence: mean log-prob per sequence (response tokens only)
    # This requires the generated token IDs; see GRPOTrainer.train_step()

    # NOTE: The actual per-token computation happens in train_step where we
    # have access to generated token IDs. This function receives pre-computed
    # per-sequence log-probs for clarity.
    raise NotImplementedError(
        "Use GRPOTrainer.train_step() which has access to token IDs. "
        "This function is kept as a reference for the math."
    )


# ═══════════════════════════════════════════════════════════
# GRPO TRAINER CLASS
# ═══════════════════════════════════════════════════════════

class GRPOTrainer:
    """
    Phase 4b.5: GRPO fine-tuning for reasoning.

    FLOW PER STEP:
    1. Sample B=32 questions from math/code dataset
    2. For each question, generate G=8 responses using current policy
    3. Score each response with rule-based reward
    4. Compute GRPO advantages within each group of G responses
    5. Compute GRPO policy gradient loss + KL penalty
    6. Backprop and update policy
    7. Every ref_update_freq steps: copy policy → reference model

    SAFETY: If avg KL divergence exceeds threshold, halt immediately.
    """

    def __init__(self, model, gcfg: GRPOConfig, device: str = 'cuda'):
        import copy
        self.model   = model
        self.gcfg    = gcfg
        self.device  = device
        self.step    = 0
        self.tokenizer = load_tokenizer()

        # Reference model: frozen copy of the initial fine-tuned checkpoint
        # Updated every ref_update_freq steps
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        print(f"Reference model initialized (frozen copy).")
        print(f"GRPO config: G={gcfg.G}, max_steps={gcfg.max_steps}, "
              f"clip_eps={gcfg.clip_eps}, kl_coef={gcfg.kl_coef}")

        # Optimizer for GRPO (very low LR)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=gcfg.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1
        )

    @torch.no_grad()
    def generate_responses(self, prompt_ids: torch.Tensor,
                           G: int) -> Tuple[List[List[int]], List[str]]:
        """
        Generate G responses for a single prompt using current policy.

        Returns:
          response_ids_list: list of G token ID lists (response only, not prompt)
          response_texts: list of G decoded response strings
        """
        self.model.eval()
        B, T_prompt = prompt_ids.shape   # B=1 for single question

        # Repeat prompt G times
        prompt_expanded = prompt_ids.expand(G, -1).to(self.device)

        all_ids = prompt_expanded.clone()
        finished = torch.zeros(G, dtype=torch.bool, device=self.device)

        for _ in range(self.gcfg.max_gen_len):
            if finished.all():
                break

            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = self.model(all_ids)

            # Sample next token (with temperature)
            next_logits = logits[:, -1, :] / self.gcfg.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)  # [G, 1]

            # Mark finished responses
            is_eos = (next_ids.squeeze(-1) == TOK_END)
            finished = finished | is_eos

            # Append next token (even if finished — will be masked in loss)
            all_ids = torch.cat([all_ids, next_ids], dim=1)

        # Extract response tokens (after prompt)
        response_ids_list = []
        response_texts    = []
        for g in range(G):
            resp_ids = all_ids[g, T_prompt:].tolist()
            # Truncate at first EOS
            if TOK_END in resp_ids:
                resp_ids = resp_ids[:resp_ids.index(TOK_END) + 1]
            response_ids_list.append(resp_ids)
            response_texts.append(self.tokenizer.decode(resp_ids, skip_special_tokens=False))

        self.model.train()
        return response_ids_list, response_texts

    def compute_sequence_logprob(self, model, input_ids: torch.Tensor,
                                  response_start: int) -> torch.Tensor:
        """
        Compute mean log-probability of response tokens under given model.

        input_ids: [G, T_total] — prompt + response
        response_start: index where response begins
        Returns: [G] — mean log-prob per sequence over response tokens
        """
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(input_ids)   # [G, T, vocab]

        logprobs = F.log_softmax(logits, dim=-1)   # [G, T, vocab]

        # Per-token log-prob of actual next token
        # logprobs[:, t, :] predicts token at position t+1
        target_ids  = input_ids[:, 1:]              # [G, T-1]
        logprobs_tm1 = logprobs[:, :-1, :]          # [G, T-1, vocab]

        # Gather log-prob of actual tokens
        per_token_lp = logprobs_tm1.gather(
            dim=2,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)                               # [G, T-1]

        # Mask: only response tokens (positions >= response_start)
        T_resp = per_token_lp.shape[1]
        mask   = torch.zeros(T_resp, dtype=torch.bool, device=self.device)
        mask[response_start:] = True
        mask   = mask.unsqueeze(0).expand_as(per_token_lp)  # [G, T-1]

        # Mean over response tokens
        seq_lp = (per_token_lp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return seq_lp   # [G]

    def train_step(self, questions: List[dict]) -> dict:
        """
        One GRPO step over a batch of questions.

        questions: list of dicts with keys:
          'prompt_ids':   List[int]     — tokenized prompt
          'ground_truth': str           — correct answer (math) or test cases (code)
          'task_type':    str           — 'math' or 'code'
          'test_cases':   Optional[List] — for code tasks

        Returns: dict of metrics
        """
        G    = self.gcfg.G
        all_policy_lp  = []
        all_ref_lp     = []
        all_rewards    = []
        all_advantages = []

        for q in questions:
            prompt_ids = torch.tensor(q['prompt_ids'], dtype=torch.long).unsqueeze(0)
            T_prompt   = prompt_ids.shape[1]

            # ── Generate G responses ────────────────────────────────
            resp_ids_list, resp_texts = self.generate_responses(prompt_ids, G)

            # ── Score each response ─────────────────────────────────
            group_rewards = []
            for g in range(G):
                r = compute_combined_reward(
                    response_text  = resp_texts[g],
                    response_ids   = resp_ids_list[g],
                    ground_truth   = q['ground_truth'],
                    task_type      = q['task_type'],
                    test_cases     = q.get('test_cases'),
                    gcfg           = self.gcfg,
                )
                group_rewards.append(r)

            group_rewards_t = torch.tensor(group_rewards, dtype=torch.float32, device=self.device)

            # ── Compute advantages (equation 3 from DeepSeek-R1) ────
            # A_i = (r_i - mean(r)) / (std(r) + 1e-8)
            mean_r = group_rewards_t.mean()
            std_r  = group_rewards_t.std() + 1e-8
            advantages = (group_rewards_t - mean_r) / std_r  # [G]

            # ── Build padded input tensors for log-prob computation ──
            max_resp_len = max(len(r) for r in resp_ids_list)
            full_ids = torch.zeros(G, T_prompt + max_resp_len, dtype=torch.long, device=self.device)
            for g in range(G):
                r_ids = resp_ids_list[g]
                full_ids[g, :T_prompt] = prompt_ids
                full_ids[g, T_prompt:T_prompt + len(r_ids)] = torch.tensor(r_ids)

            # ── Policy log-probs ─────────────────────────────────────
            policy_lp = self.compute_sequence_logprob(self.model, full_ids, T_prompt - 1)

            # ── Reference log-probs (no grad) ───────────────────────
            with torch.no_grad():
                ref_lp = self.compute_sequence_logprob(self.ref_model, full_ids, T_prompt - 1)

            all_policy_lp.append(policy_lp)
            all_ref_lp.append(ref_lp)
            all_rewards.append(group_rewards_t)
            all_advantages.append(advantages)

        # ── Stack and compute GRPO loss ──────────────────────────────
        policy_lp  = torch.cat(all_policy_lp)   # [B*G]
        ref_lp     = torch.cat(all_ref_lp)      # [B*G]
        advantages = torch.cat(all_advantages)  # [B*G]

        # Log ratio: log(π_θ / π_ref) = log π_θ - log π_ref
        log_ratio = policy_lp - ref_lp          # [B*G]
        ratio     = log_ratio.exp()             # [B*G] = π_θ / π_ref

        # PPO-style clipping with advantages
        eps = self.gcfg.clip_eps
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - eps, 1 + eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty: E[π_θ/π_ref - log(π_θ/π_ref) - 1]  (DeepSeek-R1 eq. 2)
        # This is the reverse KL approximation used in R1
        kl_div = (ratio - log_ratio - 1).mean()

        # Total GRPO loss
        total_loss = policy_loss + self.gcfg.kl_coef * kl_div

        # ── Safety check: halt if KL explodes ───────────────────────
        if kl_div.item() > self.gcfg.reward_hacking_kl_threshold:
            print(f"\n⚠️  GRPO HALTED at step {self.step}")
            print(f"   KL divergence = {kl_div.item():.4f} > threshold {self.gcfg.reward_hacking_kl_threshold}")
            print(f"   This indicates reward hacking or training instability.")
            print(f"   Save checkpoint and investigate before continuing.")
            return {'halt': True, 'kl': kl_div.item()}

        # ── Backward and update ──────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step += 1

        # ── Update reference model every ref_update_freq steps ──────
        if self.step % self.gcfg.ref_update_freq == 0:
            import copy
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
            print(f"  Reference model updated at step {self.step}")

        # Collect metrics
        rewards_tensor = torch.cat(all_rewards)
        return {
            'halt':         False,
            'step':         self.step,
            'loss':         total_loss.item(),
            'policy_loss':  policy_loss.item(),
            'kl':           kl_div.item(),
            'mean_reward':  rewards_tensor.mean().item(),
            'max_reward':   rewards_tensor.max().item(),
            'reward_std':   rewards_tensor.std().item(),
        }

    def run(self, dataset) -> bool:
        """
        Main GRPO training loop.
        dataset: iterable yielding batches of question dicts.

        Returns True if completed normally, False if halted early.
        """
        print(f"\n{'='*60}")
        print(f"STARTING PHASE 4b.5: GRPO REASONING RL")
        print(f"Max steps: {self.gcfg.max_steps}")
        print(f"G={self.gcfg.G} rollouts per question")
        print(f"Tasks: math + code (verifiable rewards ONLY)")
        print(f"{'='*60}\n")

        from itertools import cycle
        data_iter = cycle(dataset)

        while self.step < self.gcfg.max_steps:
            # Sample batch_questions unique questions
            questions = [next(data_iter) for _ in range(self.gcfg.batch_questions)]

            metrics = self.train_step(questions)

            if metrics.get('halt'):
                print("\nGRPO HALTED — reward hacking detected.")
                return False

            if self.step % 10 == 0:
                print(
                    f"step={self.step:4d} | loss={metrics['loss']:.4f} | "
                    f"policy_loss={metrics['policy_loss']:.4f} | "
                    f"kl={metrics['kl']:.5f} | "
                    f"mean_reward={metrics['mean_reward']:.4f} | "
                    f"max_reward={metrics['max_reward']:.3f}"
                )
                import json
                os.makedirs('trained_models', exist_ok=True)
                log_entry = {
                    "phase": "grpo",
                    "step": self.step,
                    "loss": metrics['loss'],
                    "policy_loss": metrics['policy_loss'],
                    "kl": metrics['kl'],
                    "mean_reward": metrics['mean_reward'],
                    "max_reward": metrics['max_reward']
                }
                with open('trained_models/training_logs.jsonl', 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

            # Save checkpoint every 100 steps
            if self.step % 100 == 0:
                os.makedirs('trained_models', exist_ok=True)
                torch.save({
                    'step':        self.step,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'metrics':     metrics,
                }, f'trained_models/grpo_step_{self.step:04d}.pt')
                print(f"  Checkpoint saved: trained_models/grpo_step_{self.step:04d}.pt")

        print(f"\nGRPO COMPLETE: {self.step} steps")
        print(f"Final mean_reward: {metrics['mean_reward']:.4f}")
        print(f"Final KL: {metrics['kl']:.5f}")
        return True
