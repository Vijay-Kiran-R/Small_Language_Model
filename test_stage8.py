# Verify CoT dataset download and format
from datasets import load_dataset

def test_stage8_cot_format():
    # OpenR1-Math-220K — primary CoT dataset
    ds = load_dataset('open-r1/OpenR1-Math-220k', split='train', streaming=True)
    example = next(iter(ds))
    print("OpenR1 fields:", list(example.keys()))
    # Should have 'problem', 'solution' or similar with <think>...</think> markers
    assert any('<think>' in str(v) for v in example.values()), \
        "OpenR1-Math-220K: no <think> tags found"
    print("STAGE 8 TEST GATE PASSED ✓  (CoT format verified)")

if __name__ == '__main__':
    test_stage8_cot_format()
