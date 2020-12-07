# Furcate
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.org/project/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)]

A lightweight wrapper for automatically forking deep learning sessions to enable parallel model training across multiple GPUs. 

## Requirements 

* Python 3.x

## Installation 

```bash 
git clone https://github.com/mattstruble/furcate.git
cd furcate && python setup.py install
```

## Usage 

Furcate allows the user to define a configuration json file which is automatically split up into every possible permutation
based on keys with array values. Below showcases an example configuration file:

```
{
    'data_name': "example", 
    'data_dir': "foo/", 
    'batch_size': [32, 64], 
    'epochs': 50
}
```

Furcate will process the above json configuration and split it up into two separate configurations, one where 
`batch_size=32` and one where `batch_size=64`. Furcate will then query the host machine for the available CUDA GPUs and 
create a queue of configurations for each GPU to query until all the permutations have been calculated. 

Simply inherit from `furcate.Fork`, implement the virtual methods, pass the config to your class and then call run.  

```python 
import furcate 

class App(furcate.Fork):
    # override virtual Fork functions

if __name__=="__main__":
    app = App("config.json")
    app.run()
```

#### Persistence 

Furcate offers persistence between runs by saving configuration and run outputs to `[log_dir]/run_data.csv`. If Furcate
detects this file on startup it will ignore any already existing runs when going through the configuration permuatations.

#### Live Updates 

While running Furcate periodically checks the configuration file for any changes. If any changes are detected Furcate 
will automatically recalculate the permutations while making sure to exclude already completed calculations. This can be
used to add, or remove, permutations without needing to restart Furcate.  

## Output 

Furcate will create a folder within the log directory per configuration, containing all of the thread's output, 
and a `run_data.csv` file which holds all the configurations and run data in csv format. 
The generated folder names are based on the provided `data_name`, `data_dir`, and a shortened `{key}{value}` 
grouping for any configuration key with multiple values. The final folder name has the following structure 
`[data_name]{thread_id}_[data_dir]_["_{shortened-key}{value}"..N]`

Below outlines the structure after running the above example:

```
logs/
run_data.csv
|-- example0_foo_bs32/
    | -- accuracy.jpg
    | -- example0.err
    | -- example0.log 
    | -- history.json
    | -- model.h5
    | -- run_data.json
|-- example1_foo_bs64/
    | -- accuracy.jpg
    | -- example1.err
    | -- example1.log 
    | -- history.json
    | -- model.h5
    | -- run_data.json
```

## Examples 

* [tensorflow](/examples/tensorflow)
    * Shows implementation of [ForkTF](#forktf) with custom dataset generation and custom configurations. 

## API

### Fork

Subclassing `furcate.Fork` will allow your app to inherit all the auto-forking capabilities and gives access to the forked configuration 
data as class member variables. For full customization Fork provides virtual methods for each stage of the model training 
pipeline each method is listed below by call-order. 

* **get_datasets(train_fp, test_fp, valid_fp)**
    * Return the computed datasets for model ingestion. The train_fp, test_fp, and valid_fp are arrays containing filepaths
    based on the provided `data_dir` value combined with the `train_prefix`, `test_prefix` and `valid_prefix` configurations.
    
* **get_model()**
    * Return the built model for use in training and evaluation

* **get_metrics()**
    * Return a list of metrics for use in training evaluation. 

* **get_optimizer()**
    * Return an optimizer for use in model compilation. 
    
* **get_loss()**
    * Return the loss for use in model compilation. 

* **model_compile(model, optimizer, loss, metrics)**
    * Passes in the built model, optmizer, loss, and metrics, to compile the model with prior to training.

* **get_callbacks()** 
    * Return a list of training callbacks. Defaults to None. 

* **model_fit(model, train_dataset, epochs, valid_dataset, callbacks, verbose)**
    * Train the model with the results from the above methods. 

* **model_evaluate(model, test_dataset)**
    * Called only if test_dataset exists. Returns the results of the model evaluation. 

* **save_metric(results_dict, history, metric)**
    * Intended to be used to save metric data to a dictionary which will then be saved to disk. It is passed a results dictionary that already contains run times,
    the history of the training session, and the current metric to save. 

* **plot_metric(history, metric)**
    * Intended to be used to save a plot of each metric to the log directory. Extract the epochs, train_metric, and val_metric
    from the history and then pass the information to `plot` to automatically save a matplotlib graph to the log dir. 

Fork also includes optional virtual functions for data preprocessing:

* **preprocess(self, record)**

* **train_preprocess(self, record)**

* **test_preprocess(self, record)**

* **valid_preprocess(self, record)**

### ForkTF

ForkTF overrides the following Fork methods with common TensorFlow implementations:

* **model_compile**
    * `model.compile(optimizer=optimizer, loss=loss, metrics=metrics)`
* **model_fit**
    * `model.fit(train_set, epochs=epochs, validation_data=valid_set, callbacks=callbacks, verbose=verbose)`
* **model_evaluate** 
    * `model.evaluate(test_set)`
* **save_metric**
    * Gets train and validation values from `history.history` and plots them individually against epochs. 
* **plot_metric**
    * Gets train and validation values from `history.history` and stores the final epoch values in the dictionary. 
    
ForkTF also comes with additional helper methods:

* **gen_tfrecord_dataset(self, filepaths, processor, shuffle=False)**
    * Creates a `TFRecordDataset` from the provided filepaths, mapped to the provided processor, 
    based on the config values for `cache`, `shuffle_buffer_size`, `batch_size`, `seed`, `prefetch`, 
    `num_parallel_reads`, and `num_parallel_calls`. 

## Configuration 

Furcate reads in configuration information from any standard json file. This allows the user to define any configuration variables
and then have them be accessible directly as Fork member variables at runtime.

Below showcases the configuration with the required configuration keys, and any reserved keys with default values:
'data_name', 'data_dir', 'batch_size', 'epochs'
```
{
  "data_name":      # String. Required. Single-word shortname describing the data being trained on. 
  "data_dir":       # String. Required. Base directory where the data lives. 
  "batch_size":     # Int.    Required. Batch size for datasets. 
  "epochs":         # Int.    Required. Number of epochs to train on. 
  "framework":      # String. Required. Deep learning framework to use, options: ['tf']. 

  "log_dir":        # String. Defaults to: "logs/"
  "train_prefix":   # String. Defaults to: "[data_name].train"
  "valid_prefix":   # String. Defaults to: "[data_name].valid"
  "test_prefix":    # String. Defaults to: "[data_name].test"
  "learning_rate":  # Float.  Defaults to: 0.001
  "verbose":        # Int.    Defaults to: 2
  "cache":          # Bool.   Defaults to: False
  "seed"            # Int.    Defaults to: 42
  "prefetch"        # Int.    Defaults to: 1 

  "meta": {
    "allow_cpu":       # Bool.   Defaults to: False
    "max_threads":     # Int.    Defaults to: Number of CUDA GPUs
    "exclude_configs": # Array.  Defaults to: []
  }
}
```

All configurations can be overwritten, and any additional configurations can be added.

#### meta.exclude_configs:

`exclude_configs` expects an array of dictionary key-value pairs and excludes any config containing those pairs from 
the generated configs. In the below example the final generated configs won't have any configuration in which both 
`optimizer_name=Adadelta` and `batch_size=32`, or `optimizer_name=SGD` and `learning_rate=0.001`.

```
{
  'data_name': "example", 
  'data_dir': "foo/", 
  'batch_size': [32, 64], 
  'epochs': 50

  "learning_rate": [0.0001, 0.001, 0.00001],
  "optimizer_name": ["RMSProp", "SGD", "Adam", "Nadam", "Adadelta"],
  "meta": {
    "exclude_configs": [
      { "optimizer_name": "Adadelta", "batch_size": 32 },
      { "optimizer_name": "SGD", "learning_rate": 0.001 }
    ]
  }
}
```

Every value in an `exclude_config` dictionary needs to be matched in order for a generated config to be excluded. 
Therefore, a dictionary pairing will need to be created per unique combination to exclude.  

## Roadmap 

* PyTorch implementation 
