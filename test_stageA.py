from slm_project.config import GRPOConfig

def test_stageA():
    gcfg = GRPOConfig()
    assert gcfg.G               == 8
    assert gcfg.kl_coef         == 0.001
    assert gcfg.max_steps       == 700
    assert gcfg.ref_update_freq == 400
    # Reward weights must sum to 1.0
    assert abs(gcfg.accuracy_weight + gcfg.format_weight + gcfg.language_weight - 1.0) < 1e-6
    print("STAGE A PASSED ✓")

if __name__ == '__main__':
    test_stageA()
