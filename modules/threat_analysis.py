"""
modules/threat_analysis.py
===========================
Rule-based Threat Analysis Module for Wildlife Camouflage Detection.

Architecture Position:
    Input Frame/Video -> U-Net -> Optical Flow -> Fusion -> YOLOv8 -> DeepSORT -> ZoeDepth -> [Threat Analysis] -> SAFE / WARNING / CRITICAL

Paper/Review Compliance:
    - Purely rule-based, fully configurable.
    - All rules, thresholds, and animal categories reside in config or are parameterised here.
    - Distance estimation is marked as:
      "Depth available (Relative: Near/Medium/Far) — Calibrated distance reserved for real-time deployment"
      and is NOT used as an arbitrary fake metric distance.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config


@dataclass
class ThreatResult:
    """Structured threat analysis output for a single object or frame."""
    threat_level: str           # "SAFE" | "WARNING" | "CRITICAL" | "UNKNOWN / LOW CONFIDENCE"
    threat_score: float         # 0.0 to 1.0
    animal_class: str           # e.g. "snow_leopard", "crab", "UNKNOWN"
    confidence: float           # Detection / segmentation confidence [0, 1]
    track_id: Optional[int]     # DeepSORT Track ID (None in single image mode)
    motion_magnitude: float     # Optical flow motion magnitude [0, 1]
    motion_level: str           # "LOW" | "MODERATE" | "HIGH"
    relative_depth: str         # "Near" | "Medium" | "Far" | "Not Available"
    depth_note: str             # Explanatory note regarding distance validation
    reason: str                 # Human-readable rule explanation
    color_bgr: Tuple[int, int, int]  # Visualization color in BGR


class RuleBasedThreatAnalyzer:
    """
    Evaluates wildlife detection, tracking, camouflage, motion, and depth
    to produce deterministic, transparent threat classifications.
    """

    # BGR Color scheme for clear display
    COLORS = {
        "SAFE": (40, 200, 40),       # Green
        "WARNING": (0, 180, 240),    # Amber/Yellow
        "CRITICAL": (30, 30, 230),   # Red
        "UNKNOWN / LOW CONFIDENCE": (160, 160, 160),  # Neutral Gray
    }

    def __init__(
        self,
        motion_low_thresh: float = config.THREAT_MOTION_LOW,
        motion_high_thresh: float = config.THREAT_MOTION_HIGH,
        conf_critical_thresh: float = config.THREAT_CONF_CRITICAL,
        persistence_critical_thresh: int = config.THREAT_PERSISTENCE_CRITICAL,
    ) -> None:
        self.motion_low_thresh = motion_low_thresh
        self.motion_high_thresh = motion_high_thresh
        self.conf_critical_thresh = conf_critical_thresh
        self.persistence_critical_thresh = persistence_critical_thresh

    def assess_threat(
        self,
        animal_class: str,
        confidence: float,
        motion_magnitude: float = 0.0,
        track_id: Optional[int] = None,
        track_persistence: int = 1,
        seg_coverage: float = 0.0,
        depth_info: Optional[Dict[str, Any]] = None,
    ) -> ThreatResult:
        """
        Evaluate an individual detected animal or camouflaged entity.
        """
        animal_lower = animal_class.lower().strip()

        # Classify motion magnitude (N/A for single image mode)
        if motion_magnitude is None or motion_magnitude < 0:
            motion_level = "N/A"
            motion_magnitude = 0.0
        elif motion_magnitude >= self.motion_high_thresh:
            motion_level = "HIGH"
        elif motion_magnitude >= self.motion_low_thresh:
            motion_level = "MODERATE"
        else:
            motion_level = "LOW"

        # Depth information parsing (relative, not fake metric)
        if depth_info and depth_info.get("available", False):
            relative_depth = depth_info.get("relative", "Medium")
            depth_note = "Relative Depth Map: AVAILABLE (Calibrated metric distance disabled in current offline prototype)"
        else:
            relative_depth = "Not Available"
            depth_note = "Depth analysis: reserved for real-time calibrated implementation"

        reasons = []

        # Safe unknown / low-confidence class handling
        # Rule: Treat "camouflaged_wildlife", "unknown" and low-confidence detections as unclassified.
        # An unclassified detection must NEVER become CRITICAL solely because of movement or track persistence.
        UNCLASSIFIED_CLASSES = {
            "camouflaged_wildlife",
            "unknown",
            "unknown / low confidence",
            "animal",
            "unclassified",
            "wildlife",
        }
        is_unclassified = (
            ("unknown" in animal_lower)
            or ("unclassified" in animal_lower)
            or (animal_lower in UNCLASSIFIED_CLASSES)
            or (confidence < config.YOLO_UNKNOWN_THRESH)
        )

        if is_unclassified:
            level = "UNKNOWN / LOW CONFIDENCE" if motion_level in ("LOW", "N/A") else "WARNING"
            reasons.append("Target identity is unclassified (camouflaged detection / low confidence); species-specific threat rules suspended")
            if motion_level in ("HIGH", "MODERATE"):
                reasons.append(f"{motion_level} movement detected in unclassified target (threat capped at WARNING)")
            elif motion_level == "N/A":
                reasons.append("Motion: N/A (Single image mode)")
            else:
                reasons.append("Low motion / stationary")

            if track_persistence >= 4:
                reasons.append(f"Persistent track confirmed ({track_persistence} frames), but threat capped below CRITICAL for unclassified targets")

            return ThreatResult(
                threat_level=level,
                threat_score=0.20 if motion_level in ("LOW", "N/A") else 0.40,
                animal_class="UNKNOWN / LOW CONFIDENCE" if ("unknown" in animal_lower or animal_lower in UNCLASSIFIED_CLASSES) else animal_class,
                confidence=float(confidence),
                track_id=track_id,
                motion_magnitude=float(motion_magnitude),
                motion_level=motion_level,
                relative_depth=relative_depth,
                depth_note=depth_note,
                reason="; ".join(reasons),
                color_bgr=self.COLORS[level],
            )

        # Base threat score from species tiers in config.py
        is_high_risk = any(pred in animal_lower for pred in config.HIGH_THREAT_ANIMALS)
        is_medium_risk = any(med in animal_lower for med in config.MEDIUM_THREAT_ANIMALS)

        score = 0.0

        if is_high_risk:
            score += 0.50
            reasons.append(f"High-risk apex/large wildlife detected ({animal_class})")
        elif is_medium_risk:
            score += 0.30
            reasons.append(f"Medium-risk wildlife detected ({animal_class})")
        else:
            score += 0.15
            reasons.append(f"Standard/camouflaged animal detected ({animal_class})")

        # Confidence modifier
        score += min(0.20, confidence * 0.20)

        # Motion modifier
        if motion_level == "HIGH":
            score += 0.25
            reasons.append("High movement detected")
        elif motion_level == "MODERATE":
            score += 0.15
            reasons.append("Moderate movement detected")
        elif motion_level == "LOW":
            reasons.append("Low motion / stationary")
        else:
            reasons.append("Motion: N/A (Single image mode)")

        # Track persistence modifier (for video tracking)
        if track_persistence >= self.persistence_critical_thresh:
            score += 0.10
            reasons.append(f"Persistent tracking confirmed ({track_persistence} frames)")

        # Cap score
        score = float(np.clip(score, 0.0, 1.0))

        # Determine level based on explicit rule priorities:
        # Rule 1: High-risk animal with moderate/high motion or high confidence => CRITICAL
        if is_high_risk and (motion_level in ("HIGH", "MODERATE") or confidence >= self.conf_critical_thresh):
            level = "CRITICAL"
        # Rule 2: Any animal with high motion and persistent track => CRITICAL
        elif motion_level == "HIGH" and track_persistence >= 4:
            level = "CRITICAL"
        # Rule 3: High-risk animal even stationary => WARNING
        elif is_high_risk:
            level = "WARNING"
        # Rule 4: Medium-risk animal with moderate/high motion => WARNING
        elif is_medium_risk and motion_level in ("HIGH", "MODERATE"):
            level = "WARNING"
        # Rule 5: Overall score >= 0.65 => CRITICAL, >= 0.40 => WARNING, else SAFE
        elif score >= 0.65:
            level = "CRITICAL"
        elif score >= 0.40:
            level = "WARNING"
        else:
            level = "SAFE"

        return ThreatResult(
            threat_level=level,
            threat_score=score,
            animal_class=animal_class,
            confidence=float(confidence),
            track_id=track_id,
            motion_magnitude=float(motion_magnitude),
            motion_level=motion_level,
            relative_depth=relative_depth,
            depth_note=depth_note,
            reason="; ".join(reasons),
            color_bgr=self.COLORS[level],
        )

    def assess_frame(
        self,
        detections: List[Dict[str, Any]],
        seg_mask: Optional[np.ndarray] = None,
        flow_map: Optional[np.ndarray] = None,
        depth_info: Optional[Dict[str, Any]] = None,
    ) -> ThreatResult:
        """
        Aggregates frame-level threats across all detections.
        If no YOLO detections exist, evaluates U-Net camouflage mask.
        """
        # Calculate scene motion from optical flow if available
        mean_motion = float(np.mean(flow_map)) if flow_map is not None else None

        # Check U-Net camouflage coverage
        seg_coverage = 0.0
        max_seg_prob = 0.0
        if seg_mask is not None:
            seg_coverage = float((seg_mask > 0.4).mean())
            max_seg_prob = float(np.max(seg_mask))

        # If detections exist from YOLO / DeepSORT:
        if detections:
            evaluated_threats = []
            for det in detections:
                res = self.assess_threat(
                    animal_class=det.get("class_name", "Animal"),
                    confidence=det.get("confidence", 0.5),
                    motion_magnitude=det.get("motion_magnitude", mean_motion),
                    track_id=det.get("track_id"),
                    track_persistence=det.get("persistence", 1),
                    seg_coverage=seg_coverage,
                    depth_info=depth_info,
                )
                evaluated_threats.append(res)

            # Sort by threat severity: CRITICAL > WARNING > SAFE / UNKNOWN, then by score
            severity_order = {"CRITICAL": 3, "WARNING": 2, "SAFE": 1, "UNKNOWN / LOW CONFIDENCE": 1}
            evaluated_threats.sort(
                key=lambda x: (severity_order.get(x.threat_level, 0), x.threat_score),
                reverse=True
            )
            return evaluated_threats[0]

        # If no YOLO detections found above threshold:
        # Strictly return UNKNOWN with LOW CONFIDENCE - never invent a generic "Camouflaged Animal" or fake 100%
        rel_depth = depth_info.get("relative", "Not Available") if depth_info else "Not Available"
        return ThreatResult(
            threat_level="UNKNOWN / LOW CONFIDENCE",
            threat_score=0.10,
            animal_class="UNKNOWN",
            confidence=0.0,
            track_id=None,
            motion_magnitude=0.0 if flow_map is None else mean_motion,
            motion_level="N/A" if flow_map is None else ("HIGH" if mean_motion >= self.motion_high_thresh else ("MODERATE" if mean_motion >= self.motion_low_thresh else "LOW")),
            relative_depth=rel_depth,
            depth_note="Relative Depth Map: AVAILABLE (Calibrated metric distance disabled)",
            reason="No target detected with sufficient confidence by YOLOv8.",
            color_bgr=self.COLORS["UNKNOWN / LOW CONFIDENCE"],
        )
