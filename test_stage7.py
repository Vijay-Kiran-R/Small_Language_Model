from slm_project.training.finetune import build_loss_mask, TOK_ASSISTANT, TOK_END, TOK_TOOL_RES, TOK_TOOL_RES_END

def test_stage7_sft_loss_masking():
    # Test 1: Basic conversation — user=0, assistant=1
    tok_ids = [
        32001,          # <|user|>
        100, 200,       # user content
        32005,          # <|end|>
        32002,          # <|assistant|>
        300, 400,       # assistant content
        32005,          # <|end|>
    ]
    mask = build_loss_mask(tok_ids)
    assert mask[0] == 0    # <|user|>
    assert mask[1] == 0    # user content
    assert mask[2] == 0    # user content
    assert mask[3] == 0    # <|end|> in user turn
    assert mask[4] == 0    # <|assistant|> delimiter
    assert mask[5] == 1    # assistant content ← TRAIN ON THIS
    assert mask[6] == 1    # assistant content
    assert mask[7] == 1    # <|end|> in assistant turn ← model must learn to stop
    print("Test 1 PASS ✓: user=0, assistant=1, EOS in asst turn=1")

    # Test 2: Tool result block — entire block = 0 including delimiters
    tok_ids_tool = [
        32002,          # <|assistant|>
        500,            # assistant content (model generates)
        32006,          # <|tool_call|>  (model generates)
        600,            # tool call content
        32007,          # <|/tool_call|> (model generates)
        32008,          # <|tool_result|>  ← DELIMITER = 0
        700, 800,       # tool result content ← = 0
        32009,          # <|/tool_result|> ← DELIMITER = 0
        900,            # more assistant content
        32005,          # <|end|>
    ]
    mask2 = build_loss_mask(tok_ids_tool)
    assert mask2[0] == 0    # <|assistant|> delimiter
    assert mask2[1] == 1    # assistant content BEFORE tool call
    assert mask2[2] == 1    # <|tool_call|>
    assert mask2[3] == 1    # tool call content
    assert mask2[4] == 1    # <|/tool_call|>
    assert mask2[5] == 0    # <|tool_result|> delimiter ← MUST BE 0
    assert mask2[6] == 0    # tool result content ← MUST BE 0
    assert mask2[7] == 0    # tool result content ← MUST BE 0
    assert mask2[8] == 0    # <|/tool_result|> delimiter ← MUST BE 0
    assert mask2[9] == 1    # assistant content AFTER tool result
    assert mask2[10] == 1   # <|end|>
    print("Test 2 PASS ✓: tool_result entire block = 0 incl. delimiters")

    print("STAGE 7 TEST GATE PASSED ✓")

if __name__ == '__main__':
    test_stage7_sft_loss_masking()
