export PATH=/workspace/.npm-global/bin:$PATH
export CLAUDE_CONFIG_DIR=/workspace/.claude
export HF_HOME=/workspace/.cache/huggingface
export PYTHONPATH=/workspace/.pylib${PYTHONPATH:+:$PYTHONPATH}   # persistent pip deps (no torch); survives pod restarts
