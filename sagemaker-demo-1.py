import sagemaker
from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.tensorflow import TensorFlow
import boto3
import os
from pathlib import Path
import urllib.request
import json


DATA_PATH = './data'

def download_dataset():
    Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
    filename, headers = urllib.request.urlretrieve('https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv', f'{DATA_PATH}/iris_training.csv')
    return filename

if __name__ == '__main__':
    source_data_file = download_dataset()
    credentials = {
        "aws_access_key_id": "",
        "aws_secret_access_key": "",
        "region_name": "us-east-1"
    }
    boto_session = boto3.Session(**credentials)
    sagemaker_session = Session(boto_session=boto_session)
    s3object = sagemaker_session.upload_data(path=DATA_PATH, key_prefix='sagemaker-demo')

    role = get_execution_role()
    estimator = TensorFlow(
        sagemaker_session=sagemaker_session,
        entry_point="./mymodel.py",
        role=role,
        hyperparameters={
            "epochs": 50
        },
        train_instance_count=1,
        train_instance_type="ml.m4.xlarge",
        base_job_name="sagemaker-demo",
        framework_version="2.4",
        py_version="py37",
        script_mode=True,

    )
    # training
    estimator.fit(s3object)

    # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/deploying_tensorflow_serving.html
    predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m5.4xlarge')

    # option 1
    out = predictor.predict({
        'instances': [
            [5.1, 3.3, 1.7, 0.5, ],
            [5.9, 3.0, 4.2, 1.5, ],
            [6.9, 3.1, 5.4, 2.1]
        ]
    })
    print(out)

    # option 2
    client = boto_session.client('sagemaker-runtime')
    response = client.invoke_endpoint(
        EndpointName=predictor.endpoint,
        Body=json.dumps({
            "instances": [
                [5.1, 3.3, 1.7, 0.5, ],
                [5.9, 3.0, 4.2, 1.5, ],
                [6.9, 3.1, 5.4, 2.1]
            ]
        }),
        ContentType='application/json',
        Accept="application/json"
    )
    print(response['Body'].read().decode())







