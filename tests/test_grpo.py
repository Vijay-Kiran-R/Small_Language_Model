import sys
sys.path.insert(0, 'slm_project')
from slm_project.training.grpo_trainer import (
    accuracy_reward_math, format_reward, language_consistency_reward,
    compute_combined_reward
)

def test_D_reward_functions():
    # ── Test accuracy_reward_math ─────────────────────────────────
    assert accuracy_reward_math("The answer is \\boxed{42}", "42") == 1.0
    assert accuracy_reward_math("The answer is \\boxed{41}", "42") == 0.0
    assert accuracy_reward_math("I think it's 42", "42") == 0.0
    assert accuracy_reward_math("\\boxed{3.14159}", "3.14159") == 1.0
    assert accuracy_reward_math("\\boxed{3.14}", "3.14159") == 0.0

    # ── Test format_reward ────────────────────────────────────────
    TOK_THINK     = 32003
    TOK_THINK_END = 32004
    TOK_END       = 32005

    ids_full_think = [32002, TOK_THINK, 100, 200, TOK_THINK_END, 300, TOK_END]
    ids_open_only  = [32002, TOK_THINK, 100, 200, TOK_END]
    ids_no_think   = [32002, 100, 200, TOK_END]
    ids_wrong_order = [TOK_THINK_END, TOK_THINK, 100, TOK_END]

    assert format_reward(ids_full_think) == 1.0
    assert format_reward(ids_open_only)  == 0.5
    assert format_reward(ids_no_think)   == 0.0
    assert format_reward(ids_wrong_order) == 0.0

    # ── Test language_consistency_reward ─────────────────────────
    pure_english = "This is the answer: 42. Machine learning models work well."
    mixed = "This is 机器学习 fine-tuning 训练."
    all_cjk = "这是答案四十二机器学习模型训练效果很好。"
    assert language_consistency_reward(pure_english, 'en') == 1.0
    assert language_consistency_reward(mixed, 'en') < 1.0
    assert language_consistency_reward(all_cjk, 'en') < 0.1

    # ── Test combined reward ──────────────────────────────────────
    from slm_project.config import GRPOConfig
    gcfg = GRPOConfig()

    perf_text = "<|think|> Let me solve this step by step. 6×7=42. </|think|> \\boxed{42}"
    perf_ids  = [TOK_THINK, 100, 200, TOK_THINK_END, 300, TOK_END]
    perfect_r = compute_combined_reward(perf_text, perf_ids, "42", "math", gcfg=gcfg)
    assert perfect_r > 0.8

    wrong_text = "<|think|> I think it's 41. </|think|> \\boxed{41}"
    wrong_ids  = [TOK_THINK, 100, TOK_THINK_END, 300, TOK_END]
    wrong_r = compute_combined_reward(wrong_text, wrong_ids, "42", "math", gcfg=gcfg)
    assert wrong_r < 0.4
