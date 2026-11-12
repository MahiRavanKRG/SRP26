#!/usr/bin/env bash
# Bigmama Cluster Sync Script
set -e

USERNAME="${1}"
HOSTNAME="master.ismll.de"
REMOTE_DIR="~/SRP26"

if [ -z "$USERNAME" ]; then
    read -p "Enter your Bigmama / Uni Hildesheim username: " USERNAME
fi

# Sanitize username
USERNAME=$(echo "$USERNAME" | tr -d '<>' | cut -d'@' -f1)

echo "=========================================================="
echo " Syncing SRP26 Bigmama Deployment to $USERNAME@$HOSTNAME"
echo "=========================================================="

echo "Creating remote directories on Bigmama..."
ssh "$USERNAME@$HOSTNAME" "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/results/cluster_results $REMOTE_DIR/logs"

echo "Uploading deployment package..."
scp -r bigmama_deployment/* "$USERNAME@$HOSTNAME:$REMOTE_DIR/"

echo "Uploading datasets..."
scp data_sets/ETTh1.csv data_sets/ETTh2.csv data_sets/weather.csv data_sets/traffic.csv data_sets/electricity.csv "$USERNAME@$HOSTNAME:$REMOTE_DIR/data/"

echo "=========================================================="
echo " Sync Complete! To launch on Bigmama:"
echo "   ssh $USERNAME@$HOSTNAME"
echo "   cd $REMOTE_DIR"
echo "   bash setup_bigmama_env.sh"
echo "   sbatch submit_array_job.sh"
echo "=========================================================="

