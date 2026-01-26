"""
Simple Airflow DAG - ML Model Training with KubernetesPodOperator
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# Default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'ml_training_dag',
    default_args=default_args,
    description='Train ML model in Kubernetes',
    schedule='@daily',
    catchup=False,
) as dag:

    # Hello World - Test task
    hello_world = KubernetesPodOperator(
        task_id='kubernetes_task',
        name='my-pod',
        namespace='airflow',
        image='python:3.11-slim',
        cmds=['python', '-c'],
        arguments=['print("Hello from Kubernetes Pod!")'],
        get_logs=True,
        on_finish_action='keep_pod',
        in_cluster=True
    )

    # ML Model Training - Using custom Docker image (RECOMMENDED)
    train_model = KubernetesPodOperator(
        task_id='train_ml_model',
        name='ml-training-pod',
        namespace='default',
        image='churnguard-trainer:latest',  # Your ML image
        get_logs=True,
        on_finish_action='keep_pod',
        in_cluster=True,
        # Important for ML workloads - set proper resources
        container_resources=k8s.V1ResourceRequirements(
            requests={'memory': '2Gi', 'cpu': '1000m'},
            limits={'memory': '4Gi', 'cpu': '2000m'},
        ),
    )

    # Task dependency
    hello_world >> train_model
