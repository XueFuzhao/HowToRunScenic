# HowToRunScenic

This repo is to help the Google Cloud Users to run one awesome JAX-based computer vision research repo, i.e. [Scenic](https://github.com/google-research/scenic)

## Prepare the code

We need some small modifications to make Scenic more friendly to the Google Cloud Users. The Scenic repo always uses the default dataset dir. However, for Google Cloud Users or other GPU users, the dataset location may be different. For instance, the Google Cloud Users usually store their dataset on Google Cloud Storage Budget. Therefore, we should edit the dataset loading function in Scenic. We use the Mnist dataset as example here. 

Before all, please fork the Scenic repo. And the you can edit your forked scenic repo.

Frist, edit the get_dataset() function in [dataset_lib/mnist_dataset.py](https://github.com/google-research/scenic/blob/main/scenic/dataset_lib/mnist_dataset.py)

1.  remove:
```
del dataset_configs
```

2. pass dataset_configs.data_dir to dataset_builder:
```
  train_ds, train_ds_info = dataset_utils.load_split_from_tfds(
      'mnist',
      batch_size,
      split='train',
      data_dir=dataset_configs.data_dir, # Added by us.
      preprocess_example=preprocess_ex,
      shuffle_seed=shuffle_seed
)
```

3. do similar thing for eval_ds:

```
  eval_ds, _ = dataset_utils.load_split_from_tfds(
      'mnist', eval_batch_size,
      split='test', 
      data_dir=dataset_configs.data_dir,
      preprocess_example=preprocess_ex)
)
```

4.  add data_dir in config file, we assume we want to use [scenic/projects/baselines/configs/mnist/mnist_config.py](https://github.com/XueFuzhao/scenic/blob/main/scenic/projects/baselines/configs/mnist/mnist_config.py) for later training:

```
  # Dataset.
  config.dataset_name = 'mnist'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.data_dir = 'YOUR_DATA_DIR'  # Added by us.
  config.data_dtype_str = 'float32'
```

Okay, the code is ready now :). For other datasets like ImageNet, we can follow the same steps as above.


## Setup the environment.

Note that all the commands in this document should be run in the commandline of
the TPU VM instance unless otherwise stated.

Please make sure you set your gcloud configs first:

1. [Create](https://console.cloud.google.com/) a GCP project.

2. [Install](https://cloud.google.com/sdk/docs/install) `gcloud`.

3. Associate your Google Account (Gmail account) with your GCP project by
   running:

   ```bash
   export GCP_PROJECT=<GCP PROJECT ID>
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project $GCP_PROJECT
   ```

4. Create a staging bucket if you do not already have one. We use europe-west4-a as an example:

   ```bash
   export GOOGLE_CLOUD_BUCKET_NAME=<GOOGLE_CLOUD_BUCKET_NAME>
   export ZONE=europe-west4-a
   gsutil mb -l $ZONE gs://$GOOGLE_CLOUD_BUCKET_NAME
   ```

Then, setup the TPU VM:

1.  Create a
    [Cloud TPU VM instance](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms)
    following
    [this instruction](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm#create-vm).
    We recommend that you develop your workflow in a single v3-8 TPU (i.e.,
    `--accelerator-type=v3-8`) and scale up to pod slices once the pipeline is
    ready. In this README, we focus on using a single v3-8 TPU. See
    [here](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) to
    learn more about TPU architectures.

2.  With Cloud TPU VMs, you ssh directly into the host machine of the TPU VM.
    You can install packages, run your code run, etc. in the host machine. Once
    the TPU instance is created, ssh into it with

    ```sh
    export TPU_NAME=v3-8
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}
    ```

    where `TPU_NAME` and `ZONE` are the name and the zone used above.


## Run your code on the TPU VM:

1.  Install the dependencies, one great way is installing t5x first. Most environment used in Scenic would be covered by that.

    ```sh
    git clone --branch=main https://github.com/google-research/t5x
    cd t5x

    python3 -m pip install -e '.[tpu]' -f \
      https://storage.googleapis.com/jax-releases/libtpu_releases.html

    ```

2. Install Scenic and Run

    ```sh
    git clone --branch=main https://github.com/YOUR_GITHUB_NAME/scenic.git
    cd scenic
    pip install .

    export WORK_DIR=gs://fuzhao/scenic/mnist
    python3 scenic/main.py \
      --config=scenic/projects/baselines/configs/mnist/mnist_config.py \
      --workdir=$WORK_DIR

    ```
## Run on multi-host TPU VMs

Sometimes, we may conduct larger scale experiments with more TPU chips (e.g. v3-128) for larger datasets like ImageNet. In this case, we can run the code in this way:

 1. Combine the code in Section ``Run your code on the TPU VM`` into a single file run_scenic.sh. We provide an example in this repo.

 2. Run

    ```sh
    export TPU_NAME=v3-128
    gcloud compute tpus tpu-vm scp run_scenic.sh $TPU_NAME: --worker=all --zone=$ZONE
    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command "bash run_t5x.sh"
    ```





