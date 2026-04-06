#!/usr/bin/env bash
echo "Update CLIENT paths if needed"

for i in {1..10}
do
nohup python3 src/client_api.py \
  --client_id client$i \
  --csv data/telemetry_client${i}.csv \
  --port 8000 \
  --win 10 \
  --class_map config/class_map.json \
  --scaler_path config/scaler.json \
  > logs/client_${i}.log 2>&1 &
done
