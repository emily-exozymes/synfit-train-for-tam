#!/bin/bash
set -e
cd /app

# SynFit source mounted at /app at runtime. Its scripts do
#   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# so importing utils.py works as long as the folder layout is right.
export PYTHONPATH=/app:/app/SynFit:${PYTHONPATH:-}

python train.py
