# CognitiveGuard (STRAND)

Real-time epistemic defense against generative montage and multi-agent belief manipulation in language models.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/tests-6%20passed-brightgreen.svg)](tests/)

---

## Overview

Recent safety evaluations have revealed that multi-agent systems and conversational language models are vulnerable to **generative montage attacks**: sequences of statements where every individual premise is factually true, but their temporal ordering and omission of connecting context reliably causes the recipient model to draw an ungrounded or false causal conclusion. Because individual tokens are non-toxic and individually factual, traditional perplexity filters and toxicity classifiers fail to detect the manipulation.

**CognitiveGuard** (STRAND: *Structured Topological Reasoning Auditing Network Defense*) is an air-gapped, zero-cost symbolic defense module that intercepts these attacks in real time. It monitors belief dynamics as evidence streams in, audits causal dependency DAGs extracted from intermediate reasoning, and blocks ungrounded state transitions before false beliefs are internalized into agent memory or emitted to users.

---

## Architecture

The system evaluates incoming multi-step information streams through two complementary mechanisms:

```
[Incoming Proposition Stream S_t]
              │
              ▼
[Kinetic Belief Tracker] ── Compute velocity v_t and acceleration a_t
              │
        (a_t > α_thresh)
              ▼
[Causal DAG Auditor]     ── Build dependency graph G = (V, E)
              │             Evaluate semantic premise support for target claim
              ▼
[Epistemic Controller]
     ├── Fully Grounded (η ≥ η_thresh) ──► Accept belief update
     └── Unsound Leap  (η < η_thresh) ──► Flag anomaly & inject counterfactual grounding prompt
```

1. **Kinetic Belief Tracker:** Computes discrete first-order velocity ($v_t = p_t - p_{t-1}$) and second-order acceleration ($a_t = v_t - v_{t-1}$) of hypothesis confidence. Sudden surges in subjective certainty without commensurate evidence volume trigger immediate auditing.
2. **Causal DAG Auditor:** Constructs a directed acyclic graph representing the reasoning chain, calculating the premise-to-claim semantic overlap ratio $\eta(G)$. If critical intermediate lemmas are missing, the step is rejected as a montage leap.

---

## Installation

```bash
git clone https://github.com/ehtesum/strand_backup.git cognitive_guard
cd cognitive_guard

# Setup environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install -e .
```

---

## Quickstart

```python
from cognitive_guard.belief_tracker import KineticBeliefTracker
from cognitive_guard.causal_dag import CausalDAGAuditor
from cognitive_guard.pipeline import CognitiveGuardPipeline

# Initialize defense pipeline
pipeline = CognitiveGuardPipeline(
    alpha_threshold=0.35,  # Maximum permitted epistemic acceleration
    eta_threshold=0.40     # Minimum required causal grounding ratio
)

# Simulated montage attack: true premises sequenced to imply an unproven causal link
adversarial_stream = [
    "Facility temperature sensors recorded an elevated thermal anomaly at 02:14.",
    "System maintenance technician Martinez clocked out of the server bay at 02:15.",
    "Therefore, Martinez intentionally disabled cooling systems to sabotage infrastructure."
]

audit_result = pipeline.process_stream(adversarial_stream)

print("Attack Intercepted:", audit_result.is_manipulation_detected)
print("Epistemic Acceleration:", audit_result.peak_acceleration)
print("Causal Grounding Score:", audit_result.grounding_score)
print("Action Taken:", audit_result.mitigation_action)
```

---

## Repository Structure

```
cognitive_guard/
├── cognitive_guard/
│   ├── __init__.py
│   ├── belief_tracker.py       # First- and second-order epistemic kinematics
│   ├── causal_dag.py           # Dependency DAG parser and premise grounding auditor
│   ├── pipeline.py             # End-to-end defense coordinator
│   └── models.py               # Data structures for propositions, states, and audits
├── data/
│   └── sample_trajectories.json # Benchmark event trajectories (benign vs adversarial)
├── examples/
│   └── run_sample_defense.py   # Minimal standalone demonstration script
├── results/
│   ├── Figure1_MontageAttack_Defense.png # Trajectory tracking under attack
│   └── Figure2_Ablation_Components.png   # Ablation across kinetic vs DAG modules
├── tests/
│   ├── test_belief_tracker.py  # Kinematics velocity and acceleration tests
│   ├── test_causal_dag.py      # DAG grounding and premise resolution tests
│   └── test_pipeline.py        # End-to-end integration tests
├── run_benchmark.py            # Reproducible benchmark runner
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

---

## Benchmark Results

Evaluated across synthetic multi-agent information streams comparing unprotected base models against CognitiveGuard:

| Defense Setting | Montage Detection Rate | False Positive Rate (Benign Feeds) | Mean Audit Overhead |
| :--- | :---: | :---: | :---: |
| Base Model (No Defense) | 0.0% (vulnerable) | — | 0.00 ms |
| Perplexity / Token Classifier | 4.2% | 12.8% | 1.82 ms |
| Secondary LLM Judge (Prompted) | 71.4% | 18.5% | 850.00 ms |
| **CognitiveGuard (Kinetic + DAG)** | **94.8%** | **2.1%** | **0.32 ms** |

---

## Running Tests

Run the full test suite with `pytest`:

```bash
pytest tests/ -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).


## 📄 Research Preprint & Publication Context

> **Academic Note:** This repository hosts the experimental implementation and evaluation harness for research currently submitted for academic conference peer review. To preserve double-blind reviewing integrity, blinded evaluation artifacts and author attributions follow standard open-access disclosure protocols.

```bibtex
@article{islam2026cognitiveguard,
  title={CognitiveGuard: Real-Time Epistemic Defense Against Belief Manipulation},
  author={Islam, Ehtesum-Ul},
  journal={arXiv preprint arXiv:2611.XXXXX},
  year={2026}
}
```
