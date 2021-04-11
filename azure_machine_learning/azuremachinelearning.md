#  Azure Machine Learning Service



![AMLS_process](https://www.blue-granite.com/hs-fs/hubfs/AMLS_process.png?width=788&name=AMLS_process.png)

![amls_site](https://www.blue-granite.com/hs-fs/hubfs/amls_site.png?width=600&name=amls_site.png)





## 0. 설치 라이브러리

- python lib 설치 requirements 파일 참조 
- python 버젼 3.6 사용 

```python
# install just the base SDK
pip install azureml-sdk

# below steps are optional
# install the base SDK, Jupyter notebook server and tensorboard
pip install azureml-sdk[notebooks,tensorboard]

# install model explainability component
pip install azureml-sdk[explain]

# install automated ml components
pip install azureml-sdk[automl]

# install experimental features (not ready for production use)
pip install azureml-sdk[contrib]


# clone the sample repoistory
git clone https://github.com/Azure/MachineLearningNotebooks.git

```



- SDK 설치 

 [SDK installation instructions](https://docs.microsoft.com/azure/machine-learning/service/how-to-configure-environment)

- 기본적인 추가 라이브러리

```
(myenv) $ conda install -y matplotlib tqdm scikit-learn
```

- ACI 등록 설정이 되어 있지 않을 경우 아래 스크립트 수행

```shell
# check to see if ACI is already registered
(myenv) $ az provider show -n Microsoft.ContainerInstance -o table
# if ACI is not registered, run this command.
# note you need to be the subscription owner in order to execute this command successfully.
(myenv) $ az provider register -n Microsoft.ContainerInstance
```



Ipython 환경 설치 및 활성화

Ipykenel 설치 및 등록을 하지 않으면 python conda 환경으로 notebook 실행 시 오류 발생 

```python
pip install ipykernel

# powershell
ipython kernel install --user --name tonne_aml --display-name "Python (myenv)"

python -m ipykernel install --user --name tonne_aml --display-name "Python (toone_aml)"
```


- 전체 환경 설정 확인 [Install the Azure Machine Learning SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python)





# 1. 추가 설치 package

```python
# scrapbook 설치 
pip install seaborn

# beautifulsoup4 설치
pip install beautifulsoup4

# scikit learn
pip install scikit-learn
```



# 2.  workspace 생성

클라우드에서 머신 러닝 모델을 실험, 교육 및 배포하는 데 사용되는 기본 리소스입니다. Azure 구독 및 리소스 그룹을 쉽게 소비되는 개체에 연결합니다.

- portal -> create resource group -> create resource -> premium 

- ***workspace -> overview 상단의 config.json 파일 다운로드***

- SDK 생성

```python
from azureml.core import Workspace

ws = Workspace.create(name=workspace_name,
                      subscription_id=subscription_id,
                      resource_group=resource_group,
                      create_resource_group=True,
                      location=workspace_region
                     )
```



- 생성한 환경 파일을 재활용 하기 위해 파일 형태로 write

```python
ws.write_config(path="./file-path", file_name="ws_config.json")
```



# 3. initialize workspace



- 직접 정보 입력 하여 진행 

```python
from azureml.core import Workspace

ws = Workspace(subscription_id="e478b470-4e14-4384-bc05-0e03fe0c2d9e",
               resource_group="HO_aml",
               workspace_name="tooneaml")

```

- 다운로드 받은 파일을 통해 접속

```python
from azureml.core import Workspace
ws = Workspace.from_config(path="../tooneaml_config.json")
```



연결 설정을 하는 방법은 4가지 존재  
로컬 상에서 진행 시에는 1번 사용  
2번은 일회성  
3, 4번은 한번 인증 후 지속적으로 활용하는 MLOps에 적합

1. Interactive Login Authentication
2. Azure CLI Authentication
3. Managed Service Identity (MSI) Authentication
4. Service Principal Authentication



# 4. experiments

각각의 모델을 수행 하는 순서 

- create experiments
- create run configuration
- create scriptrunconfig
- submit



- 실험 등록

```python
from azureml.core.experiment import Experiment

# Choose an experiment name.
experiment_name = 'aischool-text-classification-exp-01'
experiment = Experiment(ws, experiment_name)

```



- 환경 출력

```python
output = {}
output['Subscription ID'] = ws.subscription_id
output['Workspace Name'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T
```



- workspace의 실험 리스트 출력

```python
list_experiments = Experiment.list(ws)
for exp in list_experiments:
    print(exp.id, exp.name)
```



# 5. compute target 설정

- compute target은 training compute 으로 생성 

```python 
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your cluster.
amlcompute_cluster_name = "aischool-clu-01"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2", # CPU for BiLSTM, such as "STANDARD_D2_V2" 
                                                           # To use BERT (this is recommended for best performance), select a GPU such as "STANDARD_NC6" 
                                                           # or similar GPU option
                                                           # available in your workspace
                                                           max_nodes = 1)
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True)
```



- 생성 가능한 vm size 확인

```python
list_vms = AmlCompute.supported_vmsizes(workspace=ws)
```



- DSVM  생성

```python
from azureml.core.compute import ComputeTarget, RemoteCompute
from azureml.core.compute_target import ComputeTargetException

username = os.getenv('AZUREML_DSVM_USERNAME', default='<my_username>')
address = os.getenv('AZUREML_DSVM_ADDRESS', default='<ip_address_or_fqdn>')

compute_target_name = 'cpudsvm'
# if you want to connect using SSH key instead of username/password you can provide parameters private_key_file and private_key_passphrase 
try:
    attached_dsvm_compute = RemoteCompute(workspace=ws, name=compute_target_name)
    print('found existing:', attached_dsvm_compute.name)
except ComputeTargetException:
    config = RemoteCompute.attach_configuration(username=username,
                                                address=address,
                                                ssh_port=22,
                                                private_key_file='./.ssh/id_rsa')
    attached_dsvm_compute = ComputeTarget.attach(ws, compute_target_name, config)
    
    attached_dsvm_compute.wait_for_completion(show_output=True)
```



- [Compute target](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-target)

| Training  targets                                            | [Automated ML](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml) | [ML pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines) | [Azure Machine Learning designer](https://docs.microsoft.com/en-us/azure/machine-learning/concept-designer) |
| :----------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [Local computer](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#local) |                             yes                              |                                                              |                                                              |
| [Azure Machine Learning compute cluster](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#amlcompute) |                             yes                              |                             yes                              |                             yes                              |
| [Azure Machine Learning compute instance](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance) |                             yes                              |                             yes                              |                                                              |
| [Remote VM](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#vm) |                             yes                              |                             yes                              |                                                              |
| [Azure Databricks](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-your-first-pipeline#databricks) |                  yes (SDK local mode only)                   |                             yes                              |                                                              |
| [Azure Data Lake Analytics](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-your-first-pipeline#adla) |                                                              |                             yes                              |                                                              |
| [Azure HDInsight](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#hdinsight) |                                                              |                             yes                              |                                                              |
| [Azure Batch](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#azbatch) |                                                              |                             yes                              |                                                              |



# 6. runconfiguration, scriptrunconfig, run

### 1. Runconfiguration 설정 

- runconfiguration (compute_config) 설정

```python
from azureml.core import ScriptRunConfig, RunConfiguration

compute_config = RunConfiguration()

# Attach compute target to run config
compute_config.target = "amlcompute"
# runconfig.run_config.target = "local"

# compute_config.amlcompute.vm_size = "STANDARD_D1_V2"

from azureml.core.conda_dependencies import CondaDependencies

conda_dep = CondaDependencies()
#conda_dep.add_pip_package("scikit-learn")
#conda_dep.add_conda_package("numpy==1.17.0")
#conda_dep.add_pip_package("scikit-learn")
conda_dep.add_pip_package("nltk")
conda_dep.add_pip_package("pandas")
conda_dep.add_pip_package("matplotlib")

#conda_dep = CondaDependencies(conda_dependencies_file_path='./environment.yml', _underlying_structure=None)

compute_config.environment.python.conda_dependencies = conda_dep
```



- environment를 적용하여 runconfiguration 설정

```python
from azureml.core import ScriptRunConfig, RunConfiguration
from azureml.core.environment import Environment

aienv = Environment.from_existing_conda_environment(name = "aienv",
                                                    conda_environment_name = "toone_aml"

run_confg = RunConfiguration()

# Attach compute target to run config
run_confg.target = compute_target
# runconfig.run_config.target = "local"

# compute_config.amlcompute.vm_size = "STANDARD_D1_V2"


aienv.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])

run_confg.environment = aienv
```



### 2. scriptrunconfig 설정

- train 스크립트와 runconfiguration 으로  runconfiguration 설정

```python
from azureml.core import ScriptRunConfig, RunConfiguration

# run a trial from the train.py code in your current directory
src = ScriptRunConfig(source_directory='./', script='train.py',
    run_config=RunConfiguration())

```



### 3. Run 객체

런은 실험의 단일 시도를 나타냅니다. expriments 를 제출하면서 해당 실험을 run 객체로 받게 됩니다. 

이후 Run 객체를 통해 비동기 적으로 진행 되는 실행을 모니터링 하며 결과를 분석 기록 합니다. 

- 실험 실행 전 run 객체 생성 시 logging 시작

```python
run =  experiment.start_logging()
```



- 실험 실행 환경을 제출 하면서 해당 실험을 run 객체로 받음

```python
tags = {"prod": "phase-1-model-tests"}
run = experiment.submit(config=src, tags=tags)
```



- run의 세부 적인 사항 확인

```python
run_details = run.get_details()
```



- widgets을 통해 run 상태 확인 

```python
from azureml.widgets import RunDetails  
RunDetails(run).show()
```



- run list 조회 

```python
list_runs = experiment.get_runs()
for run in list_runs:
    print(run.id, run.name)
```



- run cancel

```python
run.cancel()
print(run.get_status())
```



- logging 시작 (실험 추적 )

```python
run =  experiment.start_logging()
run.log(name="message", value="Hello from run!")

```





# 7. model registry

모델 등록을 사용하여 작업 영역에서 Azure 클라우드에 모델을 저장하고 버전을 지정할 수 있습니다. 등록 된 모델은 이름과 버전으로 식별됩니다. 기존 모델과 이름이 같은 모델을 등록 할 때마다 레지스트리가 버전을 증가시킵니다. Azure Machine Learning은 Azure Machine Learning 모델뿐만 아니라 Python 3을 통해로드 할 수있는 모든 모델을 지원합니다.



- model 생성 후 outputs 폴더에 저장 하면 AML로 자동으로 업로드 됨

```python
model_file_name = 'aischool_model.pkl'
with open(model_file_name, "wb") as file:
    joblib.dump(value=pipeline, filename=os.path.join('./outputs/', model_file_name))
```

- Run 객체를 베이스로 모델 등록이 이루어지게됨

```python
model = run.register_model(model_name = "aischool_model", model_path = "./outputs/aischool_model.pkl", tags={'area': 'aichool'})
```

- 생성되어 있는 모델을 통해 등록

```python
# Register model
model = Model.register(workspace = ws,
                        model_path ="mnist/model.onnx",
                        model_name = "onnx_mnist",
                        tags = {"onnx": "demo"},
                        description = "description",)

```

- 모델을 로컬로 다운로드 

```python
from azureml.core.model import Model
import os

model = Model(workspace=ws, name="churn-model-test")
model.download(target_dir=os.getcwd())
```



### Explain Models

- Azure 에서 제공하는 interpret SDK 패키지


```python
# MS 에서 지원 하는 기능이 포함된 패키지
pip install azureml-interpret
# 미리 보기 및 실험 기능들을 사용가능
pip install azureml-contrib-interpret

# auto ml 용 
```



Azure ML interpret는 MS가 진행하는 오픈 소스 커뮤니티 interpret ML 의 소스를 사용 하였습니다. 

https://github.com/interpretml/interpret-community/

| Interpretability Technique                     | Description                                                  | Type           |
| :--------------------------------------------- | :----------------------------------------------------------- | :------------- |
| SHAP Tree Explainer                            | [SHAP](https://github.com/slundberg/shap)'s tree explainer, which focuses on polynomial time fast SHAP value estimation algorithm specific to **trees and ensembles of trees**. | Model-specific |
| SHAP Deep Explainer                            | Based on the explanation from SHAP, Deep Explainer "is a high-speed approximation algorithm for SHAP values in deep learning models that builds on a connection with DeepLIFT described in the [SHAP NIPS paper](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions). **TensorFlow** models and **Keras** models using the TensorFlow backend are supported (there is also preliminary support for PyTorch)". | Model-specific |
| SHAP Linear Explainer                          | SHAP's Linear explainer computes SHAP values for a **linear model**, optionally accounting for inter-feature correlations. | Model-specific |
| SHAP Kernel Explainer                          | SHAP's Kernel explainer uses a specially weighted local linear regression to estimate SHAP values for **any model**. | Model-agnostic |
| Mimic Explainer (Global Surrogate)             | Mimic explainer is based on the idea of training [global surrogate models](https://christophm.github.io/interpretable-ml-book/global.html) to mimic blackbox models. A global surrogate model is an intrinsically interpretable model that is trained to approximate the predictions of **any black box model** as accurately as possible. Data scientists can interpret the surrogate model to draw conclusions about the black box model. You can use one of the following interpretable models as your surrogate model: LightGBM (LGBMExplainableModel), Linear Regression (LinearExplainableModel), Stochastic Gradient Descent explainable model (SGDExplainableModel), and Decision Tree (DecisionTreeExplainableModel). | Model-agnostic |
| Permutation Feature Importance Explainer (PFI) | Permutation Feature Importance is a technique used to explain classification and regression models that is inspired by [Breiman's Random Forests paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) (see section 10). At a high level, the way it works is by randomly shuffling data one feature at a time for the entire dataset and calculating how much the performance metric of interest changes. The larger the change, the more important that feature is. PFI can explain the overall behavior of **any underlying model** but does not explain individual predictions. | Model-agnostic |





- `TabularExplainer`아래의 세 가지 SHAP 해설자 중 하나를 호출 ( `TreeExplainer`, `DeepExplainer`또는 `KernelExplainer`).
- `TabularExplainer` 사용 사례에 가장 적합한 것을 자동으로 선택하지만 기본 설명자를 각각 직접 호출 할 수 있습니다.

```python
from interpret.ext.blackbox import TabularExplainer

# "features" and "classes" fields are optional
explainer = TabularExplainer(model, 
                             x_train, 
                             features=breast_cancer_data.feature_names, 
                             classes=classes)
```





# 8. Depoly ( ACI, AKS )

### Environment 정의

Azure Machine Learning 환경은 학습 및 스코어링 스크립트와 관련된 Python 패키지, 환경 변수 및 소프트웨어 설정을 지정합니다.

- 훈련 스크립트를 개발하십시오.
- 대규모 모델 교육을 위해 Azure Machine Learning Compute에서 동일한 환경을 재사용하십시오.
- 특정 컴퓨팅 유형에 묶이지 않고 동일한 환경으로 모델을 배포하십시오.



To deploy the model as a service, you need the following components:

- **Define inference environment**. This environment encapsulates the dependencies required to run your model for inference.
- **Define scoring code**. This script accepts requests, scores the requests by using the model, and returns the results.
- **Define inference configuration**. The inference configuration specifies the environment configuration, entry script, and other components needed to run the model as a service.



- environment 생성

```python
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies


env = Environment('my-sklearn-environment')
env.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'azureml-defaults',
    'inference-schema[numpy-support]',
    'joblib',
    'numpy',
    'scikit-learn'
])
```



- environments 리스트 확인

```python
envs = Environment.list(workspace=ws)

for env in envs:
    if env.startswith("AzureML"):
        print("Name",env)
        print("packages", envs[env].python.conda_dependencies.serialize_to_string())
```



### Default environments

```python
from azureml.core import Webservice
from azureml.exceptions import WebserviceException


service_name = 'demo-07-sklearn-service'

# Remove any existing service under the same name.
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass


# 일반 배포시 default 환경으로 배포가 진행 
service = Model.deploy(ws, service_name, [model])
service.wait_for_deployment(show_output=True)
```



### Inference Configuration

- 추론 환경에 대한 정의 environment 와 entry_script를 포함 하여 설정 생성

```python
python
from azureml.core import Webservice
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException


service_name = 'my-custom-env-service'

# Remove any existing service under the same name.
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass

inference_config = InferenceConfig(entry_script='score.py', environment=environment)

```



- Deploy target

| Compute target                                               | Used for                      | GPU support                                                  | FPGA support                                                 | Description                                                  |
| :----------------------------------------------------------- | :---------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [Local web service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where#local) | Testing/debugging             |                                                              |                                                              | Use for limited testing and troubleshooting. Hardware acceleration depends on use of libraries in the local system. |
| [Azure Machine Learning compute instance web service](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where#notebookvm) | Testing/debugging             |                                                              |                                                              | Use for limited testing and troubleshooting.                 |
| [Azure Kubernetes Service (AKS)](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where#aks) | Real-time inference           | [Yes](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-inferencing-gpus) (web service deployment) | [Yes](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-fpga-web-service) | Use for high-scale production deployments. Provides fast response time and autoscaling of the deployed service. Cluster autoscaling isn't supported through the Azure Machine Learning SDK. To change the nodes in the AKS cluster, use the UI for your AKS cluster in the Azure portal. AKS is the only option available for the designer. |
| [Azure Container Instances](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where#aci) | Testing or development        |                                                              |                                                              | Use for low-scale CPU-based workloads that require less than 48 GB of RAM. |
| [Azure Machine Learning compute clusters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-run-step) | (Preview) Batch inference     | [Yes](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-run-step) (machine learning pipeline) |                                                              | Run batch scoring on serverless compute. Supports normal and low-priority VMs. |
| [Azure Functions](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-functions) | (Preview) Real-time inference |                                                              |                                                              |                                                              |
| [Azure IoT Edge](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where#iotedge) | (Preview) IoT module          |                                                              |                                                              | Deploy and serve ML models on IoT devices.                   |
| [Azure Data Box Edge](https://docs.microsoft.com/en-us/azure/databox-online/azure-stack-edge-overview) | Via IoT Edge                  |                                                              | Yes                                                          | Deploy and serve ML models on IoT devices.                   |





### ACI 배포

```python
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException

aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config)
service.wait_for_deployment(show_output=True)
```





### AKS 배포

```python
from azureml.core.compute import AksCompute, ComputeTarget

# Use the default configuration (can also provide parameters to customize)
prov_config = AksCompute.provisioning_configuration()

aks_name = 'my-aks-9' 
# Create the cluster
aks_target = ComputeTarget.create(workspace = ws, 
                                  name = aks_name, 
                                  provisioning_configuration = prov_config)
```



existing VNET에 배포하기  

- 실험과 관련된 저장소 부터 실제 실행을 위한 리소스( AKS AMLcompute 등) 에 대한 VNET 연결 설정   
  [Secure Azure ML experimentation and inference jobs within an Azure Virtual Network](



### Service check



- webservice 를 통해 결과 확인 

```  python
input_payload = json.dumps({
    'data': [
        [ 0.03807591,  0.05068012,  0.06169621, 0.02187235, -0.0442235,
         -0.03482076, -0.04340085, -0.00259226, 0.01990842, -0.01764613]
    ]
})

output = service.run(input_payload)

print(output)

```



# 9. Monitor data drift on models deployed to Azure Kubernetes service

- dataset 등록 
- model 등록

dataset을 같이  model 등록하게 되면 해당 dataset에 대해서 자동적으로 capture 를 하게 됨. 

- ##### AML을 통해 AKS에 배포된 모델에 입력되는 데이터를 모니터링 하여 학습 데이터와의 드리프트 계수라고하는 데이터 드리프트의 크기를 측정합니다.

- ##### 기능별로 데이터 드리프트 기여도를 측정하여 데이터 드리프트를 일으킨 기능을 나타냅니다.

- ##### 거리 측정 항목을 측정합니다. 현재 Wasserstein과 Energy Distance가 계산됩니다.

- ##### 피처 분포를 측정합니다. 현재 커널 밀도 추정 및 히스토그램.

- ##### 이메일로 데이터 드리프트에 경고를 보냅니다.


```python
# preview
pip install azureml-datadrift
```



# reference

[Python 용 Azure Machine Learning SDK 란 무엇입니까?](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)
