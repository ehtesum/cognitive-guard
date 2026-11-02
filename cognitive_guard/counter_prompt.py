"""
CognitiveGuard - Module 3: Adaptive Counter-Prompt Immunizer
Generates dynamically tailored epistemic skeptic prompts when an anomaly or causal illusion is flagged.
Compels the LLM analyst into self-correction before final report delivery.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .causal_dag import AuditReport
from .belief_tracker import TrajectoryStep


@dataclass
class ImmunizationPrompt:
    is_triggered: bool
    trigger_reasons: List[str]
    system_instruction: str
    counter_prompt: str
    target_claims_to_reconsider: List[str]


class AdaptiveCounterPromptImmunizer:
    """
    Constructs dynamic counter-prompts conditioned on the specific detected
    causal leaps and belief trajectory acceleration anomalies.
    """

    DEFAULT_SKEPTIC_PREFIX = (
        "[COGNITIVEGUARD REASONING AUDIT NOTICE]:\n"
        "An automated provenance audit has detected potential narrative montage and unsupported causal leaps "
        "in the intermediate synthesis of this information feed."
    )

    def __init__(self, strictness_level: str = "balanced"):
        """
        Args:
            strictness_level: 'conservative', 'balanced', or 'aggressive'.
        """
        self.strictness_level = strictness_level

    def generate_immunization(
        self,
        trajectory_step: Optional[TrajectoryStep] = None,
        audit_report: Optional[AuditReport] = None,
        target_hypothesis: Optional[str] = None
    ) -> ImmunizationPrompt:
        """
        Synthesize a targeted counter-prompt based on audit telemetry.
        """
        trigger_reasons = []
        target_claims = []

        # Check Module 1 (Belief Acceleration)
        if trajectory_step and trajectory_step.is_anomaly:
            trigger_reasons.append(
                f"Abrupt non-linear belief surge detected at step #{trajectory_step.step_index} "
                f"(acceleration alpha = {trajectory_step.acceleration:.3f} > threshold)."
            )

        # Check Module 2 (Causal DAG Audit)
        if audit_report and audit_report.is_manipulated:
            trigger_reasons.append(
                f"Causal provenance deficit: only {audit_report.grounded_edges}/{audit_report.total_inferred_edges} "
                f"causal links are verified by source facts (eta = {audit_report.provenance_ratio:.2f})."
            )
            for claim in audit_report.critical_unsupported_claims:
                target_claims.append(claim)

        is_triggered = len(trigger_reasons) > 0

        if not is_triggered:
            return ImmunizationPrompt(
                is_triggered=False,
                trigger_reasons=[],
                system_instruction="",
                counter_prompt="",
                target_claims_to_reconsider=[]
            )

        # Construct specific counter-prompt
        claims_bullet_list = ""
        if target_claims:
            claims_bullet_list = "Unverified causal claims detected:\n" + "\n".join(
                [f"  - {c}" for c in target_claims[:4]]
            )

        target_focus = ""
        if target_hypothesis:
            target_focus = (
                f"\nCRITICAL DIRECTIVE ON TARGET CONCLUSION:\n"
                f"The specific hypothesis '{target_hypothesis}' is under review. "
                f"Do NOT assume this hypothesis is True unless the source text explicitly and literally confirms it. "
                f"If the posts merely present circumstantial details or separate sequential events, you MUST declare "
                f"the verdict as 'false' or 'unverified' with low confidence."
            )
            if self.strictness_level == "aggressive":
                target_focus += (
                    f"\nSTRICT ADMISSIBILITY MANDATE: Any circumstantial deduction, narrative implication, "
                    f"or speculative correlation must be rejected outright. If direct verbatim proof is absent, "
                    f"your verdict MUST be 'false' with 0.0 confidence."
                )

        bullet_marker = "* "
        trigger_analysis_text = "\n".join([bullet_marker + r for r in trigger_reasons])

        counter_prompt_body = f"""{self.DEFAULT_SKEPTIC_PREFIX}

Trigger Analysis:
{trigger_analysis_text}

{claims_bullet_list}

MANDATORY SELF-CORRECTION INSTRUCTIONS:
1. Decouple Chronology from Causality: Just because Event A preceded Event B, do not assume A caused B.
2. Fact-Checking vs Reasoning Check: Every single tweet may be 100% true, but the overarching narrative connection may be an illusion created by sequencing.
3. Strict Grounding Constraint: Every causal claim in your final rationale must cite exact sentences from the source posts without speculative interpolation.
{target_focus}

Now re-evaluate the collected information carefully and generate your final verdict strictly adhering to the verified facts."""

        system_instruction = (
            "You are a rigorous, highly skeptical intelligence analyst. "
            "You strictly refuse to make speculative causal leaps or accept implied rumors without direct proof."
        )

        return ImmunizationPrompt(
            is_triggered=True,
            trigger_reasons=trigger_reasons,
            system_instruction=system_instruction,
            counter_prompt=counter_prompt_body,
            target_claims_to_reconsider=target_claims
        )
