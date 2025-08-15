#!/bin/bash
cd /home/goline/huy/quant_chat_bot/LLM_Project/
# Đảm bảo bash biết conda
source ~/anaconda3/etc/profile.d/conda.sh

# Kích hoạt môi trường conda
conda activate quant_chat_bot

echo "=== Bắt đầu crawler ==="
python -m app.run_schedule
echo "=== Xong ==="
