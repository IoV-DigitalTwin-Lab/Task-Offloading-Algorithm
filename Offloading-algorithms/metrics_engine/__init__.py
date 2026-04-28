"""
Realistic Metrics Engine for IoV MEC Task Offloading.

Provides physically grounded, academically credible metrics to replace
the C++ simulation's unreliable writeSingleResult() output.

Main entry point: metrics_engine.runner.run()
"""

from metrics_engine.engine import MetricsEngine
from metrics_engine.redis_interface import RedisInterface

__all__ = ["MetricsEngine", "RedisInterface"]
