from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'churn_prediction_training',
    default_args=default_args,
    description='Train churn model on Kubernetes',
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'churn'],
) as dag:

    train_model = KubernetesPodOperator(
        task_id='train_model',
        name='churn-train-pod',
        namespace='default',
        image='churn-train:latest',
        image_pull_policy='Never',
        cmds=["python", "train.py"],
        env_vars={
            'MLFLOW_TRACKING_URI': 'http://host.docker.internal:5000',
            'AWS_ACCESS_KEY_ID': 'minioadmin',
            'AWS_SECRET_ACCESS_KEY': 'minioadmin123',
            'MLFLOW_S3_ENDPOINT_URL': 'http://host.docker.internal:9000',
        },
        # Network configuration
        hostnetwork=False,
        # Allow access to host machine
        dns_policy='ClusterFirstWithHostNet',
        config_file='/home/airflow/.kube/config',
        in_cluster=False,
        is_delete_operator_pod=False, # Keep pod for debugging
        get_logs=True,
    )
