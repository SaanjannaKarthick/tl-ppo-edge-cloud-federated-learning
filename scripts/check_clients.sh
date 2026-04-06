#!/usr/bin/env bash
IPS=("CLIENT1_IP" "CLIENT2_IP" "CLIENT3_IP" "CLIENT4_IP" "CLIENT5_IP" "CLIENT6_IP" "CLIENT7_IP" "CLIENT8_IP" "CLIENT9_IP" "CLIENT10_IP")

for ip in "${IPS[@]}"
do
  echo "== $ip =="
  curl -s http://$ip:8000/health | head
done
