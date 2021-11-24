from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from textwrap import dedent

import sys
sys.path.insert(0, '/home/mandicluka/koopman-test')

from data_loader import load_and_check_config 

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator

import os, yaml, copy
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization


CFG_NAME = 'koopman_gen'

default_args = {
    'owner': 'mandicLuka',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'start_date': days_ago(2),
    # 'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

def train_and_save_model(model_name, dataset, train_params, push_xcom=False, **context):
    from data_loader import load_dataset
    from train_model import train_model_on_dataset
    from datetime import datetime
    dataset = load_dataset(ds)
    model, history = train_model_on_dataset(model_name, dataset, train_params)
    model.save(os.path.join(train_params["save_path"], f"{model_name}-{dataset}-{datetime.now()}"))
    if push_xcom:
        task = context["ti"]
        xcom_value = history.history
        xcom_value.update({"train_params": train_params})
        task.xcom_push(key='train_history', value=xcom_value)

def pick_best_model_and_retrain(dataset, models, **context):
    task = context['ti']
    histories = task.xcom_pull(key='train_history', task_ids=models)

    best = float('inf')
    index = -1
    for i, x in enumerate(histories):
        if x['val_loss'] < best:
            best = x['val_loss']
            index = i
    best_model = histories[index]

    train_params = best_model["train_params"]
    train_params["validation_split"] = 0
    train_and_save_model("best", dataset, train_params, push_xcom=False, **context)

with DAG(
    'koopman_train',
    default_args=default_args,
    description='Train koopman models',
    catchup=False,
    tags=['koopman'],
) as dag:

    cfg = load_and_check_config("config")

    hyperparam_combs = copy.deepcopy(cfg["hyperparam_combinations"])
    del cfg["hyperparam_combinations"]
    prev = None
    for ds in cfg["datasets"]:
        model_tasks = []
        for model_name, train_params in cfg["models"].items():
            for i, hyperparam_comb in enumerate(hyperparam_combs):
                train_params.update(hyperparam_comb)
                # add global cfg params to train params
                p = {key:x for key, x in cfg.items() if key not in ["datasets", "models"]}
                train_params.update(p)
                train_params
                op = PythonOperator(
                    task_id=f"{model_name}-{ds}-{i}",
                    python_callable=train_and_save_model,
                    provide_context=True,
                    op_kwargs={
                        'model_name': f"{model_name}-{i}",
                        'dataset': ds,
                        'train_params': train_params,
                        'push_xcom': True
                    },
                )
                model_tasks.append(op)
                if prev:
                    prev >> op
                prev = op
    
        evaluate = PythonOperator(
            task_id=f"evaluate_and_retrain_best-{ds}",
            python_callable=pick_best_model_and_retrain,
            provide_context=True,
            op_kwargs={
                'dataset': ds,
                'models': (x.task_id for x in model_tasks)
            },
        )
        # evaluate after all models done
        evaluate << model_tasks
