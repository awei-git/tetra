"""
Airflow DAG for daily data pipeline execution
Runs every day at 8 PM to fetch latest market data, economic indicators, events, and news
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


# Default arguments for the DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['data-alerts@tetra.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'data_pipeline_daily',
    default_args=default_args,
    description='Daily data ingestion pipeline',
    schedule_interval='0 20 * * *',  # Run at 8 PM daily
    catchup=False,
    tags=['data', 'daily', 'production'],
)

# Task 1: Run daily data update
daily_update = BashOperator(
    task_id='run_daily_update',
    bash_command=(
        'cd /opt/tetra && '
        'python -m src.pipelines.data_pipeline.main '
        '--mode=daily '
        '--date={{ ds }} '  # Airflow provides the execution date
        '2>&1'
    ),
    dag=dag,
)

# Task 2: Data quality notification (only runs if daily update succeeds)
def check_and_notify(**context):
    """Check data quality results and send notifications if needed"""
    # This would integrate with your notification system
    # For now, just log
    print(f"Data pipeline completed for {context['ds']}")
    # You could parse the pipeline output here and send alerts

quality_check = PythonOperator(
    task_id='quality_notification',
    python_callable=check_and_notify,
    provide_context=True,
    dag=dag,
)

# Task 3: Trigger downstream processes (optional)
trigger_analytics = BashOperator(
    task_id='trigger_analytics',
    bash_command='echo "Would trigger analytics pipeline here"',
    dag=dag,
)

# Define task dependencies
daily_update >> quality_check >> trigger_analytics


# Separate DAG for weekly backfill
backfill_dag = DAG(
    'data_pipeline_weekly_backfill',
    default_args=default_args,
    description='Weekly backfill to ensure data completeness',
    schedule_interval='0 2 * * 0',  # Run at 2 AM on Sundays
    catchup=False,
    tags=['data', 'backfill', 'maintenance'],
)

# Backfill last 7 days
weekly_backfill = BashOperator(
    task_id='run_weekly_backfill',
    bash_command=(
        'cd /opt/tetra && '
        'python -m src.pipelines.data_pipeline.main '
        '--mode=backfill '
        '--days=7 '
        '2>&1'
    ),
    dag=backfill_dag,
)


# Ad-hoc backfill DAG (manually triggered)
adhoc_backfill_dag = DAG(
    'data_pipeline_adhoc_backfill',
    default_args=default_args,
    description='Ad-hoc backfill for specific date ranges',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['data', 'backfill', 'manual'],
)

# This DAG expects params to be passed when triggered
adhoc_backfill = BashOperator(
    task_id='run_adhoc_backfill',
    bash_command=(
        'cd /opt/tetra && '
        'python -m src.pipelines.data_pipeline.main '
        '--mode=backfill '
        '--start-date={{ params.start_date }} '
        '--end-date={{ params.end_date }} '
        '{% if params.symbols %} --symbols {{ params.symbols }} {% endif %}'
        '2>&1'
    ),
    params={
        'start_date': '{{ ds }}',
        'end_date': '{{ ds }}',
        'symbols': ''
    },
    dag=adhoc_backfill_dag,
)