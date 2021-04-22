### ec2 에서 sagemaker session 생성
### [sagemaker sample code github](https://github.com/aws/amazon-sagemaker-examples)

1. private network 사용 시 필요한 endpoint

- sts endpoint
- **ecr dkr endpoint**
- sagemaker api endpoint
- **iam endpoint**


> If you wish to use your private VPC to securely bring your custom container, you also need the following:
> A VPC with a private subnet VPC endpoints for the following services:
> - Amazon Simple Storage Service (Amazon S3)
> - Amazon SageMaker
> - Amazon ECR
> - AWS Security Token Service (AWS STS)
> - CodeBuild for building Docker containers  
> [Bringing your own custom container image to Amazon SageMaker Studio notebooks](https://aws.amazon.com/ko/blogs/machine-learning/bringing-your-own-custom-container-image-to-amazon-sagemaker-studio-notebooks/)
> 


2. bucket 지정 
```
# 기본 bucket 지정 
bucket = sagemaker.Session().default_bucket()
# or
bucket = 'your_own_bucket'
```

3. trainnig test data input
```
# train data
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')

# test data
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')

```


4. trainning script
- container에서 실행될 py 스크립트를 가져 가기 때문에 반드시 if __name__=='__main__': 를 정의 하여 다른 포인트에서 실행되지 않도록 해야함 
> Because the container imports your training script, always put your training code in a main guard (if __name__=='__main__':) so that the container does not inadvertently run your training code at the wrong point in execution.


5. trainning model


``` python
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost


instance_type = "ml.m5.2xlarge"
output_path = 's3://{}/{}/{}/output'.format(bucket, prefix, 'abalone-dist-xgb')
content_type = "libsvm"
boto_session = boto3.Session(region_name=region)
session = Session(boto_session=boto_session)
script_path = 'abalone.py'
hyperparams = {
        "max_depth": "5",
        "eta": "0.2",
        ...
}

xgb_estimatorr = Estimator(
    entry_point=script_path,
    framework_version='1.2-1', # Note: framework_version is mandatory
    hyperparameters=hyperparams,
    role=role,
    instance_count=2,
    instance_type=instance_type,
    output_path=output_path)
    
train_input = TrainingInput("s3://{}/{}/{}/".format(bucket, prefix, "train"), content_type=content_type)
validation_input = TrainingInput("s3://{}/{}/{}/".format(bucket, prefix, "validation"),content_type=content_type)

xgb_estimatorr.fit({'train': train_input, 'validation': validation_input})
```

6. deploy

