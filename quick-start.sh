#!/bin/bash
set -e

echo "🚀 ChurnGuard MLOps Platform - Quick Start"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."

command -v docker >/dev/null 2>&1 || { echo -e "${RED}❌ Docker is not installed${NC}"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo -e "${RED}❌ Docker Compose is not installed${NC}"; exit 1; }
command -v kind >/dev/null 2>&1 || { echo -e "${RED}❌ Kind is not installed${NC}"; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}❌ kubectl is not installed${NC}"; exit 1; }

echo -e "${GREEN}✅ All prerequisites installed${NC}"
echo ""

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x init-multiple-databases.sh
echo -e "${GREEN}✅ Done${NC}"
echo ""

# Create directories
echo "📁 Creating required directories..."
mkdir -p airflow/dags airflow/plugins airflow/logs
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Set Airflow UID
if [ ! -f .env ]; then
    echo "⚙️  Setting up environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "AIRFLOW_UID=50000" > .env
    else
        echo "AIRFLOW_UID=$(id -u)" > .env
    fi
    echo -e "${GREEN}✅ .env file created${NC}"
else
    echo -e "${YELLOW}⚠️  .env file already exists, skipping${NC}"
fi
echo ""

# Start Docker Compose
echo "🐳 Starting Docker Compose services..."
docker-compose up -d

echo ""
echo -e "${YELLOW}⏳ Waiting for services to be healthy (this may take 2-3 minutes)...${NC}"
sleep 60

# Check service health
echo ""
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo -e "${GREEN}✅ Docker Compose services started${NC}"
echo ""

# Check if Kind cluster exists
if kind get clusters 2>/dev/null | grep -q "churnguard"; then
    echo -e "${YELLOW}⚠️  Kind cluster 'churnguard' already exists, skipping creation${NC}"
else
    echo "☸️  Creating Kind cluster..."
    kind create cluster --name churnguard --config kind-config.yaml --network churnguard-net
    
    echo ""
    echo "🔍 Verifying cluster..."
    kubectl cluster-info --context kind-churnguard
    echo ""
    echo -e "${GREEN}✅ Kind cluster created${NC}"
fi
echo ""

# Build training image
echo "🏗️  Building training Docker image..."
docker build -t churn-train:latest ./jobs
echo -e "${GREEN}✅ Training image built${NC}"
echo ""

# Load image into Kind
echo "📦 Loading image into Kind cluster..."
kind load docker-image churn-train:latest --name churnguard
echo -e "${GREEN}✅ Image loaded into Kind${NC}"
echo ""

# Final status
echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "📊 Access your services:"
echo "  • Airflow:  http://localhost:8080 (admin/admin)"
echo "  • MLflow:   http://localhost:5000"
echo "  • MinIO:    http://localhost:9001 (minioadmin/minioadmin123)"
echo ""
echo "🚀 Next steps:"
echo "  1. Wait 1-2 minutes for Airflow to fully initialize"
echo "  2. Open Airflow UI: http://localhost:8080"
echo "  3. Find the 'churn_prediction_training' DAG"
echo "  4. Toggle it ON and trigger it"
echo "  5. Check results in MLflow: http://localhost:5000"
echo ""
echo "📖 For detailed instructions, see: SETUP_GUIDE.md"
echo ""
echo "🛑 To stop everything:"
echo "  docker-compose down"
echo "  kind delete cluster --name churnguard"
echo ""
