# ChurnGuard Setup Guide

Complete step-by-step guide to set up and run the ChurnGuard MLOps platform.

## Initial Setup (First Time Only)

### 1. Make Scripts Executable

```bash
chmod +x init-multiple-databases.sh
```

### 2. Create Required Directories

```bash
mkdir -p airflow/dags airflow/plugins airflow/logs
```

### 3. Set Airflow UID (Linux/Mac)

```bash
echo "AIRFLOW_UID=$(id -u)" >> .env
```

On Windows (Git Bash):
```bash
echo "AIRFLOW_UID=50000" >> .env
```

## Starting the Platform

### Step 1: Start Docker Compose Services

```bash
# Start all services
docker-compose up -d

# Watch logs (optional)
docker-compose logs -f
```

Wait 2-3 minutes for all services to initialize.

### Step 2: Verify Services Are Running

```bash
docker-compose ps
```

Expected output - all services should show "Up" or "Up (healthy)":
- churnguard-postgres (healthy)
- churnguard-minio (healthy)
- churnguard-minio-init (Exited 0)
- churnguard-mlflow (Up)
- churnguard-airflow-init (Exited 0)
- churnguard-airflow (Up)

### Step 3: Verify MinIO Buckets

Access MinIO Console: http://localhost:9001
- Username: `minioadmin`
- Password: `minioadmin123`

Verify these buckets exist:
- ✅ `mlflow-artifacts`
- ✅ `airflow-logs`
- ✅ `data`

### Step 4: Create Kind Cluster

```bash
# Create the cluster (takes ~1-2 minutes)
kind create cluster --name churnguard --config kind-config.yaml --network churnguard-net
```

Verify cluster is ready:
```bash
kubectl cluster-info --context kind-churnguard
kubectl get nodes
```

Expected output:
```
NAME                       STATUS   ROLE           AGE   VERSION
churnguard-control-plane   Ready    control-plane  30s   v1.27.0
churnguard-worker          Ready    <none>         20s   v1.27.0
churnguard-worker2         Ready    <none>         20s   v1.27.0
```

### Step 5: Build and Load Training Image

```bash
# Build the Docker image
docker build -t churn-train:latest ./jobs

# Load into Kind cluster (takes ~30-60 seconds)
kind load docker-image churn-train:latest --name churnguard
```

Verify image is loaded:
```bash
docker exec -it churnguard-control-plane crictl images | grep churn-train
```

## Accessing the Services

### 1. Airflow Web UI
- URL: http://localhost:8080
- Username: `admin`
- Password: `admin`

Wait ~1-2 minutes after startup for the UI to be ready.

### 2. MLflow Tracking UI
- URL: http://localhost:5000
- No authentication required

### 3. MinIO Console
- URL: http://localhost:9001
- Username: `minioadmin`
- Password: `minioadmin123`

### 4. PostgreSQL Database
- Host: `localhost`
- Port: `5432`
- Username: `mlops`
- Password: `mlops123`
- Databases: `airflow`, `mlflow`

## Running Your First Training Job

### Step 1: Access Airflow UI

1. Open http://localhost:8080
2. Login with `admin` / `admin`

### Step 2: Find the DAG

1. Look for `churn_prediction_training` in the DAG list
2. If you don't see it, wait 30-60 seconds and refresh

### Step 3: Enable and Trigger the DAG

1. **Toggle the DAG ON** (switch on the left side)
2. Click the **Play button** (▶️) on the right
3. Select "Trigger DAG"

### Step 4: Monitor the Run

1. Click on the DAG name to see the graph view
2. Click on the `train_model` task to see logs
3. The task should show "running" → "success" (green)

### Step 5: Verify Results in MLflow

1. Open http://localhost:5000
2. Click on "churn-prediction" experiment
3. You should see a new run with:
   - Parameters: `n_estimators=100`, `max_depth=5`
   - Metrics: `accuracy`, `precision`, `recall`
   - Model artifact

## Daily Usage

After initial setup, to start working:

```bash
# Start Docker services
docker-compose up -d

# Verify Kind cluster exists (if not, create it)
kind get clusters

# Access UIs
# - Airflow: http://localhost:8080
# - MLflow: http://localhost:5000
# - MinIO: http://localhost:9001
```

## Common Commands

### Restart a Single Service
```bash
# Quick restart (doesn't pick up config changes)
docker-compose restart airflow

# Full recreate (picks up all changes)
docker-compose up -d --force-recreate airflow
```

### Check Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f airflow
docker-compose logs -f mlflow

# Last 50 lines
docker-compose logs --tail=50 airflow
```

### Rebuild Training Image
```bash
# After modifying jobs/train.py or jobs/requirements.txt
docker build -t churn-train:latest ./jobs
kind load docker-image churn-train:latest --name churnguard
```

### Check Kubernetes Pods
```bash
# List all pods
kubectl get pods -A

# Watch pod creation in real-time
kubectl get pods -w

# Check training pod logs (while running)
kubectl logs churn-train-pod
```

### Access PostgreSQL
```bash
# Via Docker
docker-compose exec postgres psql -U mlops -d airflow

# View tables
\dt

# View MLflow experiments
docker-compose exec postgres psql -U mlops -d mlflow -c "SELECT * FROM experiments;"

# Exit
\q
```

## Stopping the Platform

### Stop Docker Services
```bash
# Stop all services (keeps data)
docker-compose down

# Stop and remove volumes (deletes all data)
docker-compose down -v
```

### Delete Kind Cluster
```bash
kind delete cluster --name churnguard
```

### Remove Network (if needed)
```bash
docker network rm churnguard-net
```

## Troubleshooting Guide

### Issue: Airflow container keeps restarting
**Symptom**: `docker-compose ps` shows airflow constantly restarting

**Solutions**:
1. Check logs: `docker-compose logs airflow`
2. Verify AIRFLOW_UID is set: `cat .env | grep AIRFLOW_UID`
3. Recreate container: `docker-compose up -d --force-recreate airflow`

### Issue: MLflow not accessible
**Symptom**: http://localhost:5000 doesn't load

**Solutions**:
1. Check logs: `docker-compose logs mlflow`
2. Verify MLflow is listening on 0.0.0.0:
   ```bash
   docker-compose logs mlflow | grep "Listening"
   ```
   Should show: `Listening at: http://0.0.0.0:5000`
3. Restart: `docker-compose restart mlflow`

### Issue: DAG not appearing in Airflow UI
**Symptom**: `churn_prediction_training` not visible

**Solutions**:
1. Wait 30-60 seconds (Airflow scans every minute)
2. Check for syntax errors:
   ```bash
   docker-compose exec airflow python /opt/airflow/dags/churn_dag.py
   ```
3. Check Airflow scheduler logs:
   ```bash
   docker-compose logs airflow | grep -i "churn_dag"
   ```
4. Verify Kubernetes provider is installed:
   ```bash
   docker-compose exec airflow pip list | grep kubernetes
   ```

### Issue: Training pod fails to start
**Symptom**: Task shows "failed" in Airflow

**Solutions**:
1. Check if image exists in Kind:
   ```bash
   docker exec -it churnguard-control-plane crictl images | grep churn
   ```
2. Reload image if missing:
   ```bash
   kind load docker-image churn-train:latest --name churnguard
   ```
3. Check pod status:
   ```bash
   kubectl get pods
   kubectl describe pod churn-train-pod
   ```
4. Check pod logs:
   ```bash
   kubectl logs churn-train-pod
   ```

### Issue: Pod can't connect to MLflow/MinIO
**Symptom**: Pod logs show connection errors

**Solutions**:
1. Verify `host.docker.internal` works:
   ```bash
   kubectl run test --image=curlimages/curl --rm -it -- curl http://host.docker.internal:5000/health
   ```
2. For Linux, ensure Kind cluster was created with `--network churnguard-net`
3. Check if services are accessible from host:
   ```bash
   curl http://localhost:5000/health
   curl http://localhost:9000/minio/health/live
   ```

### Issue: Permission denied on init-multiple-databases.sh
**Symptom**: PostgreSQL fails to start with permission error

**Solution**:
```bash
chmod +x init-multiple-databases.sh
docker-compose down -v
docker-compose up -d
```

## Maintenance

### Clean Up Old Data
```bash
# Remove old Docker images
docker image prune -a

# Remove stopped containers
docker container prune

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune
```

### Check Resource Usage
```bash
# Docker containers
docker stats

# Kubernetes nodes
kubectl top nodes

# Kubernetes pods
kubectl top pods -A
```

### Backup Data
```bash
# Backup PostgreSQL databases
docker-compose exec postgres pg_dump -U mlops airflow > backup_airflow.sql
docker-compose exec postgres pg_dump -U mlops mlflow > backup_mlflow.sql

# Backup MinIO data (access via MinIO console and download buckets)
```

## Next Steps

After successfully running your first training job:

1. **Modify the training script** (`jobs/train.py`) to use real data
2. **Add data preprocessing** tasks to the DAG
3. **Create model evaluation** tasks
4. **Set up automated scheduling** (change `schedule_interval` in DAG)
5. **Deploy model serving** with FastAPI in Kubernetes
6. **Add monitoring** with Prometheus and Grafana

## Getting Help

If you encounter issues not covered here:

1. Check logs: `docker-compose logs -f`
2. Verify network: `docker network inspect churnguard-net`
3. Check GitHub issues for similar problems
4. Review Airflow documentation for KubernetesPodOperator

## Useful Links

- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [MLflow Docs](https://mlflow.org/docs/)
- [Kind Docs](https://kind.sigs.k8s.io/)
- [MinIO Docs](https://min.io/docs/)
- [Kubernetes Docs](https://kubernetes.io/docs/)
