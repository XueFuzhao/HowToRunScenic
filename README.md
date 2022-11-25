# HowToRunScenic

This repo is to help the Google Cloud Users to run one awesome JAX-based computer vision research repo, i.e. [Scenic](https://github.com/google-research/scenic)

## Prepare the code

We need some small modifications to make Scenic more friendly to the Google Cloud Users. The Scenic repo always uses the default dataset dir. However, for Google Cloud Users or other GPU users, the dataset location may be different. For instance, the Google Cloud Users usually store their dataset on Google Cloud Storage Budget. Therefore, we should edit the dataset loading function in Scenic. We use the Mnist dataset as example here. 

Frist, edit the get_dataset() function in [dataset_lib/mnist_dataset.py](https://github.com/google-research/scenic/blob/main/scenic/dataset_lib/mnist_dataset.py)

Step 1 remove:
```
del dataset_configs
```

Step 2 pass dataset_configs.data_dir to dataset_builder:
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


Step 3 do similar thing for eval_ds:

```
  eval_ds, _ = dataset_utils.load_split_from_tfds(
      'mnist', eval_batch_size,
      split='test', 
      data_dir=dataset_configs.data_dir,
      preprocess_example=preprocess_ex)
)
```

Step 4 add data_dir in config file, we assume we want to use [scenic/projects/baselines/configs/mnist/mnist_config.py](https://github.com/XueFuzhao/scenic/blob/main/scenic/projects/baselines/configs/mnist/mnist_config.py) for later training:

```
  # Dataset.
  config.dataset_name = 'mnist'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.data_dir = 'YOUR_DATA_DIR'  # Added by us.
  config.data_dtype_str = 'float32'
```

Okay, the code is ready now :)
