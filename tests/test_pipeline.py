"""
End-to-End Test for the Complete CognitiveGuard Unified Defense Pipeline
Validates Module 1 (Logit Acceleration) + Module 2 (Causal DAG Pruner) + Module 3 (Adaptive Immunization).
"""

import json
from pathlib import Path
from cognitive_guard import CognitiveGuardPipeline

def test_full_pipeline_under_montage_attack():
    print("\n================================================================================")
    print(">>> TEST 1: End-to-End CognitiveGuard Defense Against Charlie Hebdo Montage <<<")
    print("================================================================================")
    
    # Load actual attack plan
    attack_file = Path("data/copheme_adversarial/result_charliehebdo_h0.json")
    if not attack_file.exists():
        attack_file = Path(__file__).resolve().parent.parent / "data/copheme_adversarial/result_charliehebdo_h0.json"
    with open(attack_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    posts = data["attack_plan"]["posts"]
    target_conclusion = data["metadata"]["target_conclusion"]
    
    # Simulated victim step beliefs (spikes at step 3 / post 4)
    step_beliefs = [0.10, 0.15, 0.22, 0.65, 0.82, 0.88, 0.89]
    
    # Simulated victim preliminary reasoning with hallucinated causal jump
    preliminary_cot = (
        "Early reports indicate gunshots were fired near Charlie Hebdo offices. "
        "Because Ahmed Merabet was an unarmed police officer outside confronting the assailants, "
        "his intervention marks the first act of resistance in the attack. "
        "Therefore, official confirmation that he was among the officers killed proves that "
        "Ahmed Merabet was the first victim of the Charlie Hebdo attack."
    )
    
    # Initialize Unified Pipeline
    pipeline = CognitiveGuardPipeline(
        acceleration_threshold=0.25,
        provenance_threshold=0.65,
        strictness_level="balanced"
    )
    
    decision = pipeline.inspect_feed_and_reasoning(
        posts=posts,
        step_beliefs=step_beliefs,
        preliminary_rationale=preliminary_cot,
        target_hypothesis=target_conclusion
    )
    
    pipeline.print_decision_report(decision)
    
    assert decision.attack_detected is True, "Attack must be detected!"
    assert decision.module1_alert is True, "Module 1 must flag acceleration spike!"
    assert decision.module2_alert is True, "Module 2 must flag causal jump!"
    assert decision.immunization_triggered is True, "Module 3 counter-prompt must be triggered!"
    assert decision.execution_time_ms < 250.0, f"Latency {decision.execution_time_ms}ms exceeded 250ms target!"
    
    print(f"[SUCCESS] Defense successfully neutralized attack in {decision.execution_time_ms} ms!")


def test_full_pipeline_on_benign_feed():
    print("\n================================================================================")
    print(">>> TEST 2: End-to-End CognitiveGuard on Benign News Information Stream <<<")
    print("================================================================================")
    
    benign_posts = [
        {"post_id": "1", "text": "Gunshots were reported outside the Charlie Hebdo offices on Wednesday morning."},
        {"post_id": "2", "text": "Officer Ahmed Merabet was stationed near the scene and was fatally shot by attackers."},
        {"post_id": "3", "text": "Authorities later confirmed that two police officers were killed during the incidents."}
    ]
    
    benign_step_beliefs = [0.10, 0.14, 0.18]
    benign_cot = (
        "Gunshots were reported outside the Charlie Hebdo offices on Wednesday morning. "
        "Officer Ahmed Merabet was stationed near the scene and was fatally shot by attackers. "
        "Authorities later confirmed that two police officers were killed during the incidents."
    )
    
    pipeline = CognitiveGuardPipeline()
    decision = pipeline.inspect_feed_and_reasoning(
        posts=benign_posts,
        step_beliefs=benign_step_beliefs,
        preliminary_rationale=benign_cot,
        target_hypothesis="Ahmed Merabet was the first victim of the Charlie Hebdo attack"
    )
    
    pipeline.print_decision_report(decision)
    
    assert decision.attack_detected is False, "Benign feed must NOT be flagged as attack!"
    assert decision.immunization_triggered is False, "No counter-prompt should be injected for benign data!"
    assert decision.pruned_edges_count == 0, "No edges should be pruned for benign data!"
    assert decision.execution_time_ms < 250.0, f"Latency {decision.execution_time_ms}ms exceeded 250ms target!"
    
    print(f"[SUCCESS] Benign stream processed cleanly with 0 false alarms in {decision.execution_time_ms} ms!")


if __name__ == "__main__":
    test_full_pipeline_under_montage_attack()
    test_full_pipeline_on_benign_feed()
    print("\n[ALL COGNITIVEGUARD PIPELINE TESTS COMPLETED WITH 100% SUCCESS!]\n")
