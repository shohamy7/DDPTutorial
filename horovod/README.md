# Running distributed training with Horovod

## Introduction
In this tutorial, we will run distributed training job. We will start by training MNIST model, and then we will proceed
to run ResNet50 with elastic training. We are going to use the helm chart in the [current working directory](helm).
This Helm Chart will allow us to deploy our workstation that will help us to run our distributed training.
Also, We are going to use the example training modules provided by the horovod community to train the models.
For convenience, we added the modules we are going to run [here](training_modules). 
This tutorial we walk through all the steps you need to do in order to run your first Horovod job.
We will use GPU for the training process, but you can use CPU either.
So, lets get started!

## Installing the chart
In order to start running the training job, we first need to deploy a workstation.
We are going to use Kubernetes cluster and Helm Chart in order to deploy all the necessary resources to run 
our first Horovod job.
Let's start by covering the important things you need to know about the chart.
There are numerous arguments you need to know in order to start using the chart. 
The arguments are specified in the [values.yaml](helm/values.yaml). 
First is the `ssh` tab which contains arguments related to ssh configurations. 
Those arguments are used for connecting to the workers via ssh without a password.  
This is required for the openMPI to launch the training process at each slot in each worker.
In order to set up the ssh configuration, you need to generate an ssh private and public key.
You can do that with the following commands:
```shell
export SSH_KEY_DIR=`mktemp -d`
cd $SSH_KEY_DIR
yes | ssh-keygen -N "" -f id_rsa
```
Also, you have to make sure the `ssh.useSecret` argument is set to `true`.
The `ssh.hostKey` and `ssh.hostKeyPub` should be the `id_rsa` and `id_rsa.pub` created inside the `SSH_KEY_DIR` folder.

The second important tab is the `worker` tab which contains the related configuration of the workers in our workstation.
In this tab, we can specify the number of workers we want to use i.e. number of pods that will be created. 
We can do that by specifying the number of workers under the `worker.number` argument.  
The `worker.image` tab contains the related arguments of the image we will use in the workers' pod.
Similarly, the `driver.image` can be used to specify the driver's image.

The third important tab is the `resources` which contains the resources' definition for the driver and workers pods.
You can specify number of cpu, memory and GPU to request and limit for each driver and worker pods.

The last tab we are going to cover is the `datasetPvc` tab. This tab contains PVC related configurations.
If `datasetPvc.enabled` is set to `true`, a PVC will be created and used in the driver and workers pods.
This is useful when you want to have the dataset locally as all the workers need to get access to the dataset. 
You can put the dataset in this PVC, and each worker could access it.

For more information about the Horovod helm chart you can read in their [README.md](helm/README.md).

In order to install the chart, you can run the following command in the current working directory:
```shell
helm install horovod-ddp helm -n ddp \
--set ssh.useSecret=true \
--set ssh.hostKey=$(cat $SSH_KEY_DIR/id_rsa | sed 's/^/    /g') \
--set ssh.hostKeyPub=$(cat $SSH_KEY_DIR/id_rsa.pub | sed 's/^/    /g')
```
Make sure you are connected to your kubernetes cluster, and you generated an ssh private and public keys as specified earlier.
The helm chart should be installed in the `ddp` namespace (as specified by the `-n` argument).

## Running the MNIST model
In this phase, we will run the MNIST model inside our workstation. First, we will connect into our driver pod, and then
we will execute the `horovodrun` command in order to start the training job.
Let's start by connecting to our driver pod. You can that by executing the following command:
```shell
kubectl exec -it <driver-pod-name> -n ddp -- bash
```
Again, make sure you are connected to your Kubernetes cluster.
This command will allow you to run command inside the pod in your terminal.
Then, run the following command to execute the training job:
```shell
horovodrun -n 4 --hostfile /horovod/generated/hostfile --mpi-args "--mca orte_keep_fqdn_hostnames t --allow-run-as-root --display-map --tag-output --timestamp-output" bash -c 'python /horovod/examples/pytorch/pytorch_lightning_mnist.py'
```
And that's it! You run your first Horovod job. Wait until the model will finish training and see the results.
In the next phase, we'll train a bigger model which is ResNet50, and we'll also use the elastic training feature by Horovod.

## Running ResNet50 Model with fault-tolerant/elastic
In this phase, we are going to run the ResNet50 model with Horovod using the elasticity training feature.
First, lets cover how we can use Horovod to implement elastic training. 
In order to do that, we'll use the `--host-discovery-script` argument in order to add and remove hosts on the fly.
This argument is used by Horovod to detect changes in the hosts list and whether it should launch more training processes in new hosts.
In our case, we wrote a script that helps us detect new pods (which in our case are new hosts). 
The script is located inside the `/horovod/generated` directory, and it called `discoverHosts.sh`. 
This script uses `kubectl` commands in order to find new pods and detect the hosts dynamically.
Another arguments we should specify are `-np`, `--min-np`, `--max-np`. 
The `-np` argument is required, and it will be the starting number of process for the training jobs.
The `--min-np` & `--max-np` are defining the number of minimum and maximum processes for the training job to run.
When we don't have enough slots to have minimum processes to run, the job will wait until a timeout 
occurs (or until we provide more resources to the training job).
So, lets start!
Because we don't have kubectl install on the Horovod image, we will download it from the web. 
We can do this by running the following commands:
```shell
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl -LO "https://dl.k8s.io/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check
install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```
After you successfully download `kubectl`, make sure the discovery host script works as expected. In order to do that,
you can run the script with the following command:
```shell
/horovod/generated/discoverHosts.sh
```
You should see something like this:
```
ddp-horovod-0.ddp-horovod:1
ddp-horovod-1.ddp-horovod:1
ddp-horovod-2.ddp-horovod:1
ddp-horovod-driver-dpscg:1
```
The output of the script would be the available hosts that will run the training job.
In this tutorial, we'll use the tiny-imagenet dataset from kaggle in the following [link](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet).
We'll start by downloading the dataset, we'll use `kaggle-cli` for that:
```shell
pip install kaggle
mkdir -p ~/.kaggle
vi ~/.kaggle/kaggle.json  # Then put your kaggle credentials
cd /tmp  # You can choose a different directory to use
kaggle datasets download -d akash2sharma/tiny-imagenet
```
After we downloaded the dataset, we will unzip it:
```shell
apt update
apt-get install unzip
unzip tiny-imagenet.zip
```
If the data is not in the workers, we can pass it to them with the following commands:
```shell
scp -r tiny-imagenet-200 ddp-horovod-0.ddp-horovod:/tmp &
scp -r tiny-imagenet-200 ddp-horovod-1.ddp-horovod:/tmp &
scp -r tiny-imagenet-200 ddp-horovod-2.ddp-horovod:/tmp &
```
After the data is accessible by all workers, we can start to run the training process.
The following command will start the training process:
```shell
horovodrun -np 4 --min-np 4 --max-np 4 --host-discovery-script /horovod/generated/discoverHosts.sh --mpi-args "--mca orte_keep_fqdn_hostnames t --allow-run-as-root --display-map --tag-output --timestamp-output" bash -c 'python /horovod/examples/elastic/pytorch/pytorch_imagenet_resnet50_elastic.py --train-dir=/tmp/tiny-imagenet-200/train --val-dir=/tmp/tiny-imagenet-200/val --batch-size=128 --val-batch-size=128'
```
Note that we defined the `--host-discovery-script` argument as well as the `-np` argument. 
In the following example we implemented fault-tolerant training by setting `--min-np` equals to `--max-np`. 
You can instead put different values and make the training elastic.

## Conclusion
In this tutorial, we showed how you can run distributed training with Horovod. 
We covered the arguments you need to know about the `horovodrun` command in order to run a distributed training job.
However, we focus more about how to run you training script instead of how to run it. 
You can see how to write a training script supported by Horovod with the following [link](https://horovod.readthedocs.io/en/latest/pytorch.html).
In addition, you can see the example modules with use to learn more how you can implement distribute training job.
