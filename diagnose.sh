#!/bin/bash

echo "🔍 ChurnGuard MLOps Platform - Diagnostic Tool"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Docker services
echo "📊 Docker Compose Services Status:"
echo "-----------------------------------"
docker-compose ps
echo ""

# Check if services are healthy
echo "🏥 Service Health Checks:"
echo "-------------------------"

# PostgreSQL
if docker-compose ps | grep -q "churnguard-postgres.*healthy"; then
    echo -e "${GREEN}✅ PostgreSQL: Healthy${NC}"
else
    echo -e "${RED}❌ PostgreSQL: Not healthy${NC}"
fi

# MinIO
if docker-compose ps | grep -q "churnguard-minio.*healthy"; then
    echo -e "${GREEN}✅ MinIO: Healthy${NC}"
else
    echo -e "${RED}❌ MinIO: Not healthy${NC}"
fi

# MLflow
if docker-compose ps | grep -q "churnguard-mlflow.*Up"; then
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ MLflow: Running and accessible${NC}"
    else
        echo -e "${YELLOW}⚠️  MLflow: Running but not accessible on port 5000${NC}"
    fi
else
    echo -e "${RED}❌ MLflow: Not running${NC}"
fi

# Airflow
if docker-compose ps | grep -q "churnguard-airflow.*Up"; then
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Airflow: Running and accessible${NC}"
    else
        echo -e "${YELLOW}⚠️  Airflow: Running but not accessible on port 8080${NC}"
    fi
else
    echo -e "${RED}❌ Airflow: Not running${NC}"
fi

echo ""

# Check Kind cluster
echo "☸️  Kubernetes (Kind) Cluster:"
echo "------------------------------"
if kind get clusters 2>/dev/null | grep -q "churnguard"; then
    echo -e "${GREEN}✅ Kind cluster 'churnguard' exists${NC}"
    
    # Check nodes
    echo ""
    echo "Cluster nodes:"
    kubectl get nodes 2>/dev/null || echo -e "${RED}❌ Cannot connect to cluster${NC}"
    
    # Check if training image exists
    echo ""
    echo "Training image in cluster:"
    if docker exec churnguard-control-plane crictl images 2>/dev/null | grep -q "churn-train"; then
        echo -e "${GREEN}✅ churn-train:latest image loaded${NC}"
    else
        echo -e "${RED}❌ churn-train:latest image not found${NC}"
        echo "   Run: kind load docker-image churn-train:latest --name churnguard"
    fi
else
    echo -e "${RED}❌ Kind cluster 'churnguard' not found${NC}"
    echo "   Run: kind create cluster --name churnguard --config kind-config.yaml --network churnguard-net"
fi

echo ""

# Check Docker network
echo "🌐 Network Configuration:"
echo "-------------------------"
if docker network ls | grep -q "churnguard-net"; then
    echo -e "${GREEN}✅ Docker network 'churnguard-net' exists${NC}"
else
    echo -e "${RED}❌ Docker network 'churnguard-net' not found${NC}"
fi

echo ""

# Check MinIO buckets
echo "🪣 MinIO Buckets:"
echo "----------------"
if command -v mc >/dev/null 2>&1; then
    mc alias set local http://localhost:9000 minioadmin minioadmin123 2>/dev/null
    if mc ls local 2>/dev/null | grep -q "mlflow-artifacts"; then
        echo -e "${GREEN}✅ mlflow-artifacts bucket exists${NC}"
    else
        echo -e "${RED}❌ mlflow-artifacts bucket missing${NC}"
    fi
    
    if mc ls local 2>/dev/null | grep -q "data"; then
        echo -e "${GREEN}✅ data bucket exists${NC}"
    else
        echo -e "${RED}❌ data bucket missing${NC}"
    fi
    
    if mc ls local 2>/dev/null | grep -q "airflow-logs"; then
        echo -e "${GREEN}✅ airflow-logs bucket exists${NC}"
    else
        echo -e "${RED}❌ airflow-logs bucket missing${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  MinIO client (mc) not installed, cannot verify buckets${NC}"
    echo "   Check manually at: http://localhost:9001"
fi

echo ""

# Check PostgreSQL databases
echo "🗄️  PostgreSQL Databases:"
echo "-------------------------"
if docker-compose exec -T postgres psql -U mlops -l 2>/dev/null | grep -q "airflow"; then
    echo -e "${GREEN}✅ airflow database exists${NC}"
else
    echo -e "${RED}❌ airflow database missing${NC}"
fi

if docker-compose exec -T postgres psql -U mlops -l 2>/dev/null | grep -q "mlflow"; then
    echo -e "${GREEN}✅ mlflow database exists${NC}"
else
    echo -e "${RED}❌ mlflow database missing${NC}"
fi

echo ""

# Check if training image is built
echo "🐳 Docker Images:"
echo "----------------"
if docker images | grep -q "churn-train.*latest"; then
    echo -e "${GREEN}✅ churn-train:latest image built${NC}"
else
    echo -e "${RED}❌ churn-train:latest image not found${NC}"
    echo "   Run: docker build -t churn-train:latest ./jobs"
fi

echo ""

# Recent errors in logs
echo "📋 Recent Errors in Logs:"
echo "-------------------------"
echo "Airflow errors (last 5):"
docker-compose logs --tail=100 airflow 2>/dev/null | grep -i "error" | tail -5 || echo "No recent errors"

echo ""
echo "MLflow errors (last 5):"
docker-compose logs --tail=100 mlflow 2>/dev/null | grep -i "error" | tail -5 || echo "No recent errors"

echo ""
echo ""
echo "=============================================="
echo "💡 Common Solutions:"
echo "=============================================="
echo ""
echo "If services are not healthy:"
echo "  docker-compose down && docker-compose up -d"
echo ""
echo "If Airflow is not accessible:"
echo "  docker-compose up -d --force-recreate airflow"
echo ""
echo "If training image is missing in Kind:"
echo "  docker build -t churn-train:latest ./jobs"
echo "  kind load docker-image churn-train:latest --name churnguard"
echo ""
echo "If buckets are missing:"
echo "  docker-compose restart minio-init"
echo ""
echo "For full logs of a service:"
echo "  docker-compose logs -f <service-name>"
echo ""
