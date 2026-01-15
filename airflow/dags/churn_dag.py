from airflow.sdk import dag, task
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
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

@dag(
    'churn_prediction_training',
    default_args=default_args,
    description='Train churn model on Kubernetes',
    schedule=None,
    catchup=False,
    tags=['mlops', 'churn'],
)
def churn_training_dag():

    @task()
    def k82_pod_operator():
        print("Starting churn model training DAG")
        churn_training = KubernetesPodOperator(
            namespace='mlops',
            image='churn-model-trainer:latest',
            cmds=['python', 'train_churn_model.py'],
            name='churn-model-training-pod',
            task_id='churn_model_training_task',
            get_logs=True,
            is_delete_operator_pod=True,
        )
    k82_pod_operator()

churn_training_dag()
    
