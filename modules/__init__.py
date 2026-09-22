"""modules/__init__.py — exposes Member 1 modules at package level."""
from .camouflage_processor import CamouflageProcessor, ProcessorResult
from .optical_flow         import FarnebackOpticalFlow
from .fusion               import FrameFusion

__all__ = [
    "CamouflageProcessor",
    "ProcessorResult",
    "FarnebackOpticalFlow",
    "FrameFusion",
]
