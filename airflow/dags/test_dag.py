from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'kubernetes_pod_example',
    default_args=default_args,
    description='Example DAG with KubernetesPodOperator',
    schedule=timedelta(days=1),
    catchup=False,
)

task = KubernetesPodOperator(
    task_id='kubernetes_task',
    name='my-pod',
    namespace='airflow',
    image='python:3.11-slim',
    cmds=['python', '-c'],
    arguments=['print("Hello from Kubernetes Pod!")'],
    get_logs=True,
    dag=dag,
)