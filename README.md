# Furcate
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.org/project/isort/)
[![Build Status](https://travis-ci.org/mattstruble/furcate.svg?branch=main)](https://travis-ci.org/mattstruble/furcate)
[![Coverage Status](https://coveralls.io/repos/github/mattstruble/furcate/badge.svg?branch=main)](https://coveralls.io/github/mattstruble/furcate?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
![Lint](https://github.com/mattstruble/furcate/workflows/Lint/badge.svg)

A lightweight wrapper for automatically forking deep learning sessions to enable parallel model training across multiple GPUs. 

## Requirements 

* Python 3.x
* Matplotlib
* Pandas

[furcate-tf](#furcate-tf) additionally requires `tensorflow>=2.0`

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
    * Shows implementation of [furcate-tf](#furcate-tf) with custom dataset generation and custom configurations. 

## API

### Fork

Subclassing `furcate.Fork` will allow your app to inherit all the auto-forking capabilities and gives access to the forked configuration 
data as class member variables. For full customization Fork provides virtual methods for each stage of the model training 
pipeline each method is listed below by call-order. 

* **get_available_gpu_indices()**
    * Returns a list of available GPU indices. Defaults to using `nvidia-smi`.

* **set_visible_gpus()**
    * Restricts the CUDA visible devices on disk to the current processor's GPU id. 
    Defaults to setting OS Environment Variable `CUDA_VISIBLE_DEVICES` to `self.gpu_id`. 

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
    
* **get_callbacks()** 
    * Return a list of training callbacks. Defaults to None. 

* **model_compile(model, optimizer, loss, metrics)**
    * Passes in the built model, optmizer, loss, and metrics, to compile the model with prior to training.

* **model_summary(model)**
    * Intended to be used to display the model summary to log. 

* **model_fit(model, train_dataset, epochs, valid_dataset, callbacks, verbose)**
    * Train the model with the results from the above methods. 

* **model_evaluate(model, test_dataset)**
    * Called only if test_dataset exists. Returns the results of the model evaluation. 
    
* **model_save(model)**
    * Intended to be used to save the fully trained model to disk. 

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

### furcate-tf

furcate-tf overrides the following Fork methods with common TensorFlow implementations:

* **get_available_gpu_indices**
    * Utilizes `tf.config.list_physical_devices("GPU")` to get the indices of devices visible to TensorFlow.
* **set_visible_gpus**
    * `tf.config.set_visible_devices([GPU_ID], "GPU")`
* **model_compile**
    * `model.compile(optimizer=optimizer, loss=loss, metrics=metrics)`
* **model_summary**
    * `model.summary()`
* **model_fit**
    * `model.fit(train_set, epochs=epochs, validation_data=valid_set, callbacks=callbacks, verbose=verbose)`
* **model_evaluate** 
    * `model.evaluate(test_set)`
* **model_save**
    * `model.save(os.path.join(self.log_dir, "model.h5"))`
* **save_metric**
    * Gets train and validation values from `history.history` and stores the final epoch values in the dictionary. 
* **plot_metric**
    * Gets train and validation values from `history.history` and plots them individually against epochs. 
    
furcate-tf also comes with additional helper methods:

* **gen_tfrecord_dataset(self, filepaths, processor, shuffle=False)**
    * Creates a `TFRecordDataset` from the provided filepaths, mapped to the provided processor, 
    based on the config values for `cache`, `shuffle_buffer_size`, `batch_size`, `seed`, `prefetch`, 
    `num_parallel_reads`, and `num_parallel_calls`. 
    
`num_parallel_reads` and `num_parallel_calls` can be configured like every other data value, 
but if unset furcate-tf defaults them to `tf.data.experimental.AUTOTUNE`.
 
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
  "seed":           # Int.    Defaults to: 42
  "prefetch":       # Int.    Defaults to: 1 

  "meta": {
    "allow_cpu":       # Bool.   Defaults to: False
    "max_threads":     # Int.    Defaults to: Number of CUDA GPUs
    "exclude_configs": # Array.  Defaults to: []
	
	"mem_trace": { 	   # Dictionary. Defaults to below configuration. 
		"enabled": 	   # Boolean. Defaults to false. 
		"delay": 	   # Int. Number of seconds between running memory trace. Defaults to: 300 
		"top":		   # Int. Number of top memory allocations to list. Defaults to: 10 
		"trace": 	   # Int. Number of of allocations to display the stack trace for. Defaults to: 1 
	}
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

#### meta.mem_trace:

`mem_trace` controls the built-in memory trace configuration, which periodically outputs memory usage statistics to the furcate.log file. It tracks the CUDA GPU stats, top memory changes since starting Furcate, top incremental changes, and the current top memory allocations. Below shows an example output of the top 3 memory allocations from a single snapshot while running a training environment:

```
2020-12-02 16:04:33.000907: furcate.runner] GPU Stats
2020-12-02 16:04:33.000907: furcate.runner] gpu_stats id=0, name=TITAN RTX, mem_used=19278 MiB, mem_total=24220 MiB, mem_util=79 %, volatile_gpu=46 %, temp=60 C
2020-12-02 16:04:33.000908: furcate.runner] gpu_stats id=1, name=TITAN RTX, mem_used=16564 MiB, mem_total=24220 MiB, mem_util=68 %, volatile_gpu=33 %, temp=54 C
2020-12-02 16:04:33.000908: furcate.runner] Top Diffs since Start
2020-12-02 16:04:33.000909: furcate.runner] top_diffs i=1, stat=XXX/python3.8/linecache.py:0: size=316 KiB (+316 KiB), count=3173 (+3173), average=102 B
2020-12-02 16:04:33.000910: furcate.runner] top_diffs i=2, stat=XXX/python3.8/tracemalloc.py:0: size=73.6 KiB (+73.6 KiB), count=1084 (+1084), average=70 B
2020-12-02 16:04:33.000910: furcate.runner] top_diffs i=3, stat=XXX/python3.8/abc.py:0: size=25.2 KiB (+25.2 KiB), count=144 (+144), average=179 B
2020-12-02 16:04:33.000910: furcate.runner] Top Incremental
2020-12-02 16:04:33.000910: furcate.runner] top_incremental i=1, stat=XXX/python3.8/linecache.py:137: size=315 KiB (+101 KiB), count=3156 (+1018), average=102 B
2020-12-02 16:04:33.000910: furcate.runner] top_incremental i=2, stat=XXX/python3.8/tracemalloc.py:185: size=25.5 KiB (+16.6 KiB), count=435 (+280), average=60 B
2020-12-02 16:04:33.000911: furcate.runner] top_incremental i=3, stat=XXX/python3.8/tracemalloc.py:532: size=12.9 KiB (+12.9 KiB), count=195 (+195), average=68 B
2020-12-02 16:04:33.000911: furcate.runner] Top Current
2020-12-02 16:04:33.000924: furcate.runner] top_current i=1, stat=XXX/python3.8/linecache.py:0: size=316 KiB, count=3173, average=102 B
2020-12-02 16:04:33.000924: furcate.runner] top_current i=2, stat=XXX/python3.8/tracemalloc.py:0: size=73.6 KiB, count=1084, average=70 B
2020-12-02 16:04:33.000924: furcate.runner] top_current i=3, stat=XXX/python3.8/abc.py:0: size=25.2 KiB, count=144, average=179 B
2020-12-02 16:04:33.000936: furcate.runner] traceback memory_blocks=2138, size_kB=213
2020-12-02 16:04:33.000936: furcate.runner]   File "dlpa_model_research/dlpa_model.py", line 191
2020-12-02 16:04:33.000936: furcate.runner]     app.run()
2020-12-02 16:04:33.000936: furcate.runner]   File "XXX/python3.8/site-packages/furcate-0.1.0.dev1-py3.8.egg/furcate/fork.py", line 87
2020-12-02 16:04:33.000936: furcate.runner]     runner.run(self.script_name)
2020-12-02 16:04:33.000936: furcate.runner]   File "XXX/python3.8/site-packages/furcate-0.1.0.dev1-py3.8.egg/furcate/runner.py", line 358
2020-12-02 16:04:33.000936: furcate.runner]     mem_trace.snapshot("Started thread {}: [{}]".format(thread_id, str(config)))
2020-12-02 16:04:33.000936: furcate.runner]   File "XXX/python3.8/site-packages/furcate-0.1.0.dev1-py3.8.egg/furcate/runner.py", line 134
2020-12-02 16:04:33.000936: furcate.runner]     for line in stat.traceback.format():
2020-12-02 16:04:33.000936: furcate.runner]   File "XXX/python3.8/tracemalloc.py", line 229
2020-12-02 16:04:33.000936: furcate.runner]     line = linecache.getline(frame.filename, frame.lineno).strip()
2020-12-02 16:04:33.000936: furcate.runner]   File "XXX/python3.8/linecache.py", line 16
2020-12-02 16:04:33.000936: furcate.runner]     lines = getlines(filename, module_globals)
2020-12-02 16:04:33.000937: furcate.runner]   File "XXX/python3.8/linecache.py", line 47
2020-12-02 16:04:33.000937: furcate.runner]     return updatecache(filename, module_globals)
2020-12-02 16:04:33.000937: furcate.runner]   File "XXX/python3.8/linecache.py", line 137
2020-12-02 16:04:33.000937: furcate.runner]     lines = fp.readlines()
```

## Roadmap 

* Hyperparameter auto-tuning
* PyTorch implementation 
