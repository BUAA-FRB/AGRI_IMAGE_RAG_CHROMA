#!/usr/bin/env bash
set -e
python .pre_work/run_agent.py --sites .pre_work/examples/sites.json --extend_to_30 --no_llm
