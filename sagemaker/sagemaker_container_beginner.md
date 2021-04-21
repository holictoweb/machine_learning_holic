### ec2 에서 sagemaker session 생성

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




