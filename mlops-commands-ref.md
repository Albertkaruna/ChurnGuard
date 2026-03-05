# MLOps Command Reference

## System Setup

```bash
# Docker (Ubuntu)
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker

# Kind
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.31.0/kind-linux-amd64
chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind

# Helm
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-4
chmod 700 get_helm.sh && ./get_helm.sh

# Kubectl
sudo snap install kubectl --classic
```

## Kind Cluster

```bash
kind create cluster --config kind-config.yaml
kind delete cluster -n churnguard
kind get clusters
kind export kubeconfig --name churnguard
kind load docker-image myimage:latest --name churnguard
```

## Helm Repos

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add apache-airflow https://airflow.apache.org
helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update
helm repo list
helm search repo nginx
helm search hub wordpress --list-repo-url
```

## Helm Chart Lifecycle

```bash
helm install <release> <chart>
helm upgrade --install <release> <chart> --namespace <ns> --create-namespace --values values.yaml
helm upgrade <release> <chart> --namespace <ns> --reuse-values --set key=value
helm uninstall <release>
helm list -A
helm history <release>
helm rollback <release> <revision>
helm status <release> --view-resources
helm get values <release>
helm get manifest <release>
helm show values <chart> > values.yaml
helm pull <chart> --untar
```

## Airflow on K8s

```bash
helm upgrade --install airflow apache-airflow/airflow \
  --namespace airflow --create-namespace --values airflow-values.yaml

kubectl port-forward svc/airflow-api-server 8080:8080 -n airflow
kubectl create secret generic my-webserver-secret \
  --from-literal="webserver-secret-key=$(python3 -c 'import secrets; print(secrets.token_hex(16))')" -n airflow

kubectl exec -n airflow deployment/airflow-scheduler -- ls -la /opt/airflow/dags
kubectl exec -n airflow deployment/airflow-scheduler -- pip list | grep kubernetes
kubectl rollout restart deployment/airflow-scheduler -n airflow
kubectl rollout restart deployment/airflow-api-server -n airflow
```

## MLflow on K8s (Bitnami)

```bash
helm upgrade --install mlflow oci://registry-1.docker.io/bitnamicharts/mlflow \
  --namespace mlflow --create-namespace --values mlflow_values.yaml \
  --set global.security.allowInsecureImages=true

kubectl port-forward svc/mlflow-tracking 5000:80 -n mlflow

# MLflow credentials
echo Username: $(kubectl get secret --namespace mlflow mlflow-tracking -o jsonpath="{ .data.admin-user }" | base64 -d)
echo Password: $(kubectl get secret --namespace mlflow mlflow-tracking -o jsonpath="{.data.admin-password }" | base64 -d)
```

## MLflow MinIO

```bash
kubectl port-forward svc/mlflow-minio 9000:9000 -n mlflow
kubectl port-forward svc/mlflow-minio 9001:9001 -n mlflow
kubectl exec -it <minio-pod> -n mlflow -- bash

# MinIO credentials
kubectl get secret mlflow-minio -n mlflow -o jsonpath="{.data.root-user}" | base64 -d
kubectl get secret mlflow-minio -n mlflow -o jsonpath="{.data.root-password}" | base64 -d
```

## Prometheus & Grafana

```bash
kubectl port-forward svc/kube-prometheus-stack-prometheus 9090:9090 -n monitoring
kubectl port-forward svc/kube-prometheus-stack-grafana 9091:80 -n monitoring

# Grafana credentials
echo $(kubectl get secret --namespace monitoring kube-prometheus-stack-grafana -o jsonpath="{.data.admin-user}" | base64 -d)
echo $(kubectl get secret --namespace monitoring kube-prometheus-stack-grafana -o jsonpath="{.data.admin-password}" | base64 -d)

kubectl get servicemonitor
kubectl apply -f service-monitor.yaml
```

## Docker Build & Load to Kind

```bash
docker build -t myimage:latest .
kind load docker-image myimage:latest --name churnguard
kubectl rollout restart deployment/mydeployment
# all-in-one
docker build -t myimage . && kind load docker-image myimage:latest --name churnguard && kubectl rollout restart deployment/mydeployment

# verify image loaded in kind
docker exec -it churnguard-control-plane crictl images | grep myimage
```

## Kubectl Essentials

```bash
kubectl get all -n <ns>
kubectl get pods -n <ns> -w                       # watch mode
kubectl get svc --all-namespaces | grep <term>
kubectl get nodes -o wide
kubectl get events -n <ns> --sort-by='.lastTimestamp'
kubectl config get-contexts
kubectl config get-clusters
```

## Pod Debugging

```bash
kubectl describe pod/<name> -n <ns>
kubectl logs pod/<name> -n <ns> -f                 # follow logs
kubectl logs pod/<name> -n <ns> --all-containers=true
kubectl exec -it <pod> -n <ns> -- bash

# Run a throwaway debug pod
kubectl run debug-pod -n <ns> --image=busybox --rm -it --restart=Never -- sh

# DNS test from inside cluster
kubectl run dns-test -n <ns> --image=busybox --rm -it --restart=Never -- nslookup myservice.mynamespace.svc.cluster.local

# Curl test from inside cluster
kubectl run curl-test -n <ns> --image=curlimages/curl --rm -it --restart=Never -- curl -v http://myservice.mynamespace.svc.cluster.local:80/health
```

## Cleanup

```bash
# Docker
docker system prune -a --volume
docker builder prune -a
docker image prune -a
docker system df -v

# Kubernetes
kubectl delete pods --all -n <ns> --force
kubectl delete pvc,jobs -n <ns> --all
kubectl delete ns <ns>
helm delete <release> --namespace <ns>
```

## API Testing (curl)

```bash
curl -X GET http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -1.2, 0.8]}'
```
