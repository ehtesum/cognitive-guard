"""
CognitiveGuard - Module 1: Belief Trajectory & Logit Acceleration Tracker
Mathematical Formulation:
    P_t = P(H_f | E_{1:t})
    Velocity: v_t = dP/dt ≈ P_t - P_{t-1}
    Acceleration: α_t = d²P/dt² ≈ P_t - 2*P_{t-1} + P_{t-2}

Anomalous montage attacks induce sharp, localized acceleration spikes (α_t > τ_α),
distinguishing manipulative generative montage from gradual, organic belief acquisition.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrajectoryStep:
    step_index: int
    post_id: Any
    belief_prob: float
    velocity: float = 0.0
    acceleration: float = 0.0
    is_anomaly: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BeliefTrajectoryTracker:
    """
    Real-time tracker for victim agent belief trajectory over sequential evidence steps.
    Monitors velocity and acceleration to detect Kuleshov montage manipulation.
    """

    def __init__(
        self,
        acceleration_threshold: float = 0.30,
        velocity_threshold: float = 0.40,
        window_size: int = 3,
        adaptive_threshold: bool = True
    ):
        """
        Args:
            acceleration_threshold: Static second-derivative threshold (default: 0.30).
            velocity_threshold: Static first-derivative threshold (default: 0.40).
            window_size: Smoothing window size.
            adaptive_threshold: Whether to scale threshold dynamically based on baseline variance.
        """
        self.acceleration_threshold = acceleration_threshold
        self.velocity_threshold = velocity_threshold
        self.window_size = window_size
        self.adaptive_threshold = adaptive_threshold
        
        self.history: List[TrajectoryStep] = []
        self.accelerations: List[float] = []

    def reset(self):
        """Reset trajectory history for a new evidence session."""
        self.history.clear()
        self.accelerations.clear()

    def update(
        self,
        belief_prob: float,
        post_id: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TrajectoryStep:
        """
        Record a new belief probability at step t and calculate velocity & acceleration.

        Args:
            belief_prob: Current posterior confidence P(H_f | E_{1:t}) in range [0.0, 1.0].
            post_id: Identifier of the post arriving at this step.
            metadata: Optional dictionary with role, timestamp, etc.

        Returns:
            TrajectoryStep containing computed metrics and anomaly flag.
        """
        step_idx = len(self.history)
        metadata = metadata or {}
        belief_prob = float(np.clip(belief_prob, 0.0, 1.0))

        # Calculate Velocity (1st derivative)
        if step_idx == 0:
            velocity = 0.0
            acceleration = 0.0
        elif step_idx == 1:
            velocity = belief_prob - self.history[0].belief_prob
            acceleration = 0.0
        else:
            p_t = belief_prob
            p_t_minus_1 = self.history[-1].belief_prob
            p_t_minus_2 = self.history[-2].belief_prob

            velocity = p_t - p_t_minus_1
            # Discrete second derivative: d²P/dt² ≈ P_t - 2*P_{t-1} + P_{t-2}
            acceleration = p_t - (2.0 * p_t_minus_1) + p_t_minus_2

        # Anomaly Detection Logic
        effective_threshold = self.acceleration_threshold
        if self.adaptive_threshold and len(self.accelerations) >= 3:
            # Dynamic z-score based adaptation
            mean_acc = float(np.mean(self.accelerations))
            std_acc = float(np.std(self.accelerations))
            if std_acc > 1e-4:
                effective_threshold = max(self.acceleration_threshold, mean_acc + 2.0 * std_acc)

        is_anomaly = False
        if step_idx >= 2:
            self.accelerations.append(acceleration)
            # Anomaly triggered if positive acceleration spike exceeds threshold
            # and velocity is strongly positive (belief surging towards falsehood)
            if acceleration > effective_threshold and velocity > 0.15:
                is_anomaly = True

        step = TrajectoryStep(
            step_index=step_idx,
            post_id=post_id if post_id is not None else step_idx,
            belief_prob=belief_prob,
            velocity=round(velocity, 4),
            acceleration=round(acceleration, 4),
            is_anomaly=is_anomaly,
            metadata=metadata
        )

        self.history.append(step)
        return step

    def get_trajectory_summary(self) -> Dict[str, Any]:
        """
        Generate summary metrics for the entire belief trajectory session.
        """
        if not self.history:
            return {"status": "empty"}

        beliefs = [s.belief_prob for s in self.history]
        velocities = [s.velocity for s in self.history]
        accelerations = [s.acceleration for s in self.history]
        anomalies = [s for s in self.history if s.is_anomaly]

        max_acc_step = max(self.history, key=lambda s: s.acceleration) if self.history else None

        return {
            "total_steps": len(self.history),
            "initial_belief": beliefs[0],
            "final_belief": beliefs[-1],
            "net_belief_shift": round(beliefs[-1] - beliefs[0], 4),
            "peak_velocity": round(max(velocities), 4) if velocities else 0.0,
            "peak_acceleration": round(max_acc_step.acceleration, 4) if max_acc_step else 0.0,
            "peak_acceleration_step": max_acc_step.step_index if max_acc_step else 0,
            "peak_acceleration_post_id": max_acc_step.post_id if max_acc_step else None,
            "anomaly_detected": len(anomalies) > 0,
            "num_anomaly_steps": len(anomalies),
            "anomaly_steps": [s.step_index for s in anomalies]
        }

    def print_trajectory_table(self):
        """Print an ASCII telemetry table of the belief trajectory."""
        print("\n" + "=" * 85)
        print("COGNITIVEGUARD: BELIEF TRAJECTORY & ACCELERATION TELEMETRY")
        print("=" * 85)
        print(f"{'Step':<6} | {'Post ID':<10} | {'Role':<22} | {'P(Hf)':<8} | {'Velocity':<10} | {'Accel(a)':<10} | {'Status'}")
        print("-" * 85)
        for s in self.history:
            role = s.metadata.get("sequencing_role", "unknown")
            status = "[ALERT SPIKE!]" if s.is_anomaly else "[OK Normal]"
            print(f"{s.step_index:<6} | {str(s.post_id):<10} | {role:<22} | {s.belief_prob:<8.3f} | {s.velocity:<10.3f} | {s.acceleration:<10.3f} | {status}")
        print("=" * 85)
        summary = self.get_trajectory_summary()
        print(f"Summary: Net Shift={summary['net_belief_shift']:+.3f} | Peak Accel={summary['peak_acceleration']} at Step #{summary['peak_acceleration_step']} | Alert={summary['anomaly_detected']}")
        print("=" * 85 + "\n")
