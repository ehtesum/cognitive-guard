"""
Validation test for CognitiveGuard Module 1 (BeliefTrajectoryTracker)
Compares a simulated Generative Montage attack trajectory against a benign news trajectory.
"""

import json
from pathlib import Path
from cognitive_guard.belief_tracker import BeliefTrajectoryTracker

def test_montage_attack_trajectory():
    print("\n>>> TEST 1: Simulating Generative Montage Attack (Charlie Hebdo h0) <<<")
    
    # Load actual attack plan
    attack_file = Path("data/copheme_adversarial/result_charliehebdo_h0.json")
    if not attack_file.exists():
        attack_file = Path(__file__).resolve().parent.parent / "data/copheme_adversarial/result_charliehebdo_h0.json"
    with open(attack_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    posts = data["attack_plan"]["posts"]
    target_conclusion = data["metadata"]["target_conclusion"]
    print(f"Target Conclusion: '{target_conclusion}'")
    print(f"Total Posts in Montage: {len(posts)}")
    
    # In the base paper attack, the victim's belief starts low,
    # remains modest during context establishment (post 1-2),
    # then violently surges during tension building and climax (post 4-6)
    # reaching high confidence (>0.80) at post 7.
    simulated_attack_beliefs = [0.10, 0.15, 0.22, 0.65, 0.82, 0.88, 0.89]
    
    tracker = BeliefTrajectoryTracker(acceleration_threshold=0.25, adaptive_threshold=True)
    
    for i, post in enumerate(posts):
        step = tracker.update(
            belief_prob=simulated_attack_beliefs[i],
            post_id=post["post_id"],
            metadata={
                "sequencing_role": post.get("sequencing_role", "unknown"),
                "text_snippet": post["text"][:50] + "..."
            }
        )
    
    tracker.print_trajectory_table()
    summary = tracker.get_trajectory_summary()
    assert summary["anomaly_detected"] is True, "Expected anomaly alert on attack trajectory!"
    print(f"[SUCCESS] Attack detected! Peak acceleration a={summary['peak_acceleration']} at Step #{summary['peak_acceleration_step']}")


def test_benign_learning_trajectory():
    print("\n>>> TEST 2: Simulating Benign / Organic Information Gathering <<<")
    # In benign learning, belief increases gradually or fluctuates mildly without sharp acceleration spikes
    benign_beliefs = [0.10, 0.14, 0.20, 0.28, 0.35, 0.41, 0.46]
    
    tracker = BeliefTrajectoryTracker(acceleration_threshold=0.25, adaptive_threshold=True)
    
    for i, b in enumerate(benign_beliefs):
        tracker.update(
            belief_prob=b,
            post_id=f"benign_{i+1}",
            metadata={"sequencing_role": "organic_news"}
        )
    
    tracker.print_trajectory_table()
    summary = tracker.get_trajectory_summary()
    assert summary["anomaly_detected"] is False, "Benign feed should NOT trigger anomaly alert!"
    print("[SUCCESS] Benign feed passed with 0 False Positives!")


if __name__ == "__main__":
    test_montage_attack_trajectory()
    test_benign_learning_trajectory()
    print("\n[ALL MODULE 1 TESTS PASSED PERFECTLY]\n")
