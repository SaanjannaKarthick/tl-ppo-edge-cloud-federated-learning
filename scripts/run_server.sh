+ #!/usr/bin/env bash
+ set -euo pipefail
+ echo "Update CLIENT1_IP to CLIENT10_IP before running"
+ mkdir -p logs outputs
+
+ nohup python3 src/server_experiment_cade.py \
+   --csv data/telemetry_merged.csv \
+   --label_col scenario \
+   --clients http://CLIENT1_IP:8000,http://CLIENT2_IP:8000,http://CLIENT3_IP:8000,http://CLIENT4_IP:8000,http://CLIENT5_IP:8000,http://CLIENT6_IP:8000,http://CLIENT7_IP:8000,http://CLIENT8_IP:8000,http://CLIENT9_IP:8000,http://CLIENT10_IP:8000 \
+   --rounds 300 \
+   --seeds 42,43,44,45,46 \
+   --strategies fedavg,random,greedy,dqn,ddqn,tl_ppo \
+   --outdir outputs/afrl_runs_cade_300r \
+   --epochs 3 \
+   --lr 0.001 \
+   --batch_size 64 \
+   --alpha 1.0 \
+   --beta 0.5 \
+   --gamma 0.3 \
+   > logs/run_cade_300r_all.log 2>&1 &