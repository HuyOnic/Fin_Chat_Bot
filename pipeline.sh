#!/bin/bash
source .env
API_BASE_URL="http://localhost:8000"
echo "[$(date)] === Bắt đầu pipeline ===" >> config/pipeline.log

# Gọi API chạy toàn bộ pipeline
curl -X POST "http://localhost:8001/run-pipeline?crawl_source=all&days=1000&threshold=0.85" >> config/pipeline.log 2>&1

echo "[$(date)] === Kết thúc pipeline ===" >> config/pipeline.log
