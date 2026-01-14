# ChurnGuard - Cloud-Native MLOps Platform

A local MLOps platform for automated churn prediction using Kubernetes, Airflow, MLflow, and MinIO.

## Architecture

### Components
- **Docker Compose**: PostgreSQL, MinIO, Airflow, MLflow
- **Kubernetes (Kind)**: Training workloads, model serving
- **Monitoring**: Prometheus & Grafana (optional)

### Network Flow
```
Airflow (Docker) → Kind (K8s) → Training Pod → MLflow (Docker) ← MinIO (Docker)
```

## Prerequisites

- Docker Desktop or Docker Engine (20.10+)
- Docker Compose (2.0+)
- kubectl (1.27+)
- Kind (0.20+)
- Minimum 16GB RAM, 4 CPUs allocated to Docker

## Quick Start

### Step 1: Make Scripts Executable

```bash
chmod +x init-multiple-databases.sh
```

### Step 2: Start Docker Compose Services

```bash
# Create airflow directories
mkdir -p airflow/dags airflow/plugins

# Start all services
docker-compose up -d

# Check status (wait 2-3 minutes)
docker-compose ps
```

### Step 3: Create Kind Cluster

```bash
kind create cluster --name churnguard --config kind-config.yaml --network churnguard-net
```

Verify:
```bash
kubectl cluster-info --context kind-churnguard
kubectl get nodes
```

### Step 4: Build and Load Training Image

```bash
# Build the training image
docker build -t churn-train:latest ./jobs

# Load image into Kind cluster
kind load docker-image churn-train:latest --name churnguard
```

Verify image is loaded:
```bash
docker exec -it churnguard-control-plane crictl images | grep churn-train
```

### Step 5: Access Services

| Service | URL | Username | Password |
|---------|-----|----------|----------|
| **Airflow** | http://localhost:8080 | admin | admin |
| **MLflow** | http://localhost:5000 | - | - |
| **MinIO Console** | http://localhost:9001 | minioadmin | minioadmin123 |
| **PostgreSQL** | localhost:5432 | mlops | mlops123 |

## Running the Training Pipeline

1. **Access Airflow UI**: http://localhost:8080
2. **Find DAG**: Look for `churn_prediction_training`
3. **Toggle On**: Enable the DAG
4. **Trigger**: Click the "Play" button to trigger manually

The DAG will:
- Create a Kubernetes pod in your Kind cluster
- Run the training script (`train.py`)
- Log metrics and model to MLflow
- Clean up the pod after completion

## Project Structure

```
ChurnGuard/
├── docker-compose.yml          # Docker Compose services
├── kind-config.yaml            # Kind cluster configuration
├── init-multiple-databases.sh  # PostgreSQL init script
├── .gitignore                  # Git ignore rules
├── airflow/
│   ├── dags/
│   │   └── churn_dag.py       # Training DAG
│   └── plugins/                # Custom Airflow plugins
├── jobs/
│   ├── train.py               # Training script
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # Training image
└── README.md
```

## Network Connectivity

### Docker Compose ↔ Kubernetes

Since both run on the same `churnguard-net` Docker network:

**From Kubernetes pods to Docker Compose:**
- MinIO: `http://host.docker.internal:9000`
- MLflow: `http://host.docker.internal:5000`
- PostgreSQL: `postgresql://mlops:mlops123@postgres:5432/mlflow`

**From Docker Compose to Kubernetes:**
- Airflow connects via `~/.kube/config` (mounted in container)

## Environment Variables

The following environment variables are used (set in docker-compose.yml):

```bash
# PostgreSQL
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops123
POSTGRES_DB=postgres
POSTGRES_MULTIPLE_DATABASES=airflow,mlflow

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__CORE__LOAD_EXAMPLES=false
AIRFLOW__CORE__FERNET_KEY=46BKJoQYlPPOexq0OhDZnIlNepKFf87WFwLbfzqDDho=
AIRFLOW__WEBSERVER__SECRET_KEY=churnguard-secret-key
AIRFLOW__API__SECRET_KEY=churnguard-secret-key
AIRFLOW_UID=1000

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123
```

## Troubleshooting

### Airflow Not Starting
```bash
# Check logs
docker-compose logs -f airflow

# Recreate container
docker-compose up -d --force-recreate airflow
```

### MLflow Connection Issues
```bash
# Test connection from training pod
docker-compose exec mlflow curl http://localhost:5000/health

# Restart MLflow
docker-compose restart mlflow
```

### Kubernetes Pod Issues
```bash
# Check pod status
kubectl get pods -A

# Check pod logs
kubectl logs <pod-name>

# Verify image exists in Kind
docker exec -it churnguard-control-plane crictl images
```

### Network Connectivity
```bash
# Test from K8s to Docker Compose
kubectl run test --image=curlimages/curl --rm -it -- curl http://host.docker.internal:5000/health

# Verify Docker network
docker network inspect churnguard-net
```

## Stopping Services

### Stop Docker Compose
```bash
docker-compose down
```

### Delete Kind Cluster
```bash
kind delete cluster --name churnguard
```

### Remove Network
```bash
docker network rm churnguard-net
```

## Next Steps

1. **Add Data Validation**: Integrate Great Expectations
2. **Deploy FastAPI**: Create model serving API in Kubernetes
3. **Add Monitoring**: Deploy Prometheus & Grafana to Kind
4. **Configure ArgoCD**: Set up GitOps for automated deployments
5. **Implement CI/CD**: Add GitHub Actions for automation

## Common Issues & Solutions

### Issue: DAG not appearing in Airflow UI
**Solution**: 
1. Check DAG file syntax: `docker-compose exec airflow python /opt/airflow/dags/churn_dag.py`
2. Wait 30-60 seconds for Airflow to parse the file
3. Check for import errors in Airflow UI (top banner)

### Issue: `ModuleNotFoundError: No module named 'airflow.providers.cncf.kubernetes'`
**Solution**: The Kubernetes provider is installed at runtime via the docker-compose command. If it fails, rebuild the container:
```bash
docker-compose up -d --force-recreate airflow
```

### Issue: Pod can't reach MLflow/MinIO
**Solution**: Verify `host.docker.internal` resolves correctly:
- On Mac/Windows: Works by default
- On Linux: May need to add `--add-host=host.docker.internal:host-gateway` to Kind config

## Resource Management

### Check Resource Usage
```bash
# Docker stats
docker stats

# Kubernetes resource usage
kubectl top nodes
kubectl top pods -A
```

### Optimization Tips
- Stop unused services: `docker-compose stop <service-name>`
- Clean old Docker images: `docker system prune -a`
- Limit Docker Desktop resources in settings

## Contributing

This is a practice/learning project. Feel free to:
- Add additional ML features
- Enhance monitoring capabilities
- Implement data drift detection
- Add A/B testing infrastructure

## License

MIT License - Free to use for learning and practice.

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Kind Documentation](https://kind.sigs.k8s.io/)
- [MinIO Documentation](https://min.io/docs/)
