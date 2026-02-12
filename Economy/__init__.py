"""
Economy module: estimate economic loss from predicted hazard labels + field_stats,
then optionally call a local Qwen model to produce expert-style analysis, and finally
generate a multimodal Markdown report with charts.

Standalone:
- Reads output/latest.json
- Reads datasets/.../field_stats.json
- Writes output/economy_out/* and output/assets/charts/*
"""

from .economy_agent import EconomyAgent, EconomyConfig
from .models import EconomyResult, EconomyItem
