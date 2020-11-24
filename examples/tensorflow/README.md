### Requirements 

Running the above example requires the following python packages to be installed:

```
furcate
wget
tensorflow > 2.x
```

### Running

To run the module simply execute the command `python3 horses_vs_humans.py`. After downloading the training data 
the example will:
 
 1. Load the config.json and generate configurations based on ever permutation of data stored in a list 
 2. Detect the number of available GPUs and set that as the maximum number of concurrent threads
    * Max threads can be hardcoded in the config as metadata: {'meta'{'max_threads':X'}}
 3. Spin up subprocesses that execute the code in the `App` class for each config permutation.
    * Each subprocess will be assigned its own GPU to train on  
 


### Example Output 

The thread runner will output INFO level logs with thread timings and an estimate remaining total time based on the
remaining number of configurations and the average run time across all completed runs:

```
2020-11-20 17:56:25.000628: furcate.runner] Couldn't find max_threads in config, defaulting to number of GPUs [2].
2020-11-20 17:59:25.000810: furcate.runner] Thread 0 finished - 02m 16s - est. total time remaining: 11m 23s
2020-11-20 17:59:25.000810: furcate.runner] Thread 1 finished - 02m 16s - est. total time remaining: 11m 23s
2020-11-20 18:02:25.000991: furcate.runner] Thread 3 finished - 02m 21s - est. total time remaining: 09m 13s
2020-11-20 18:02:25.000991: furcate.runner] Thread 2 finished - 02m 16s - est. total time remaining: 09m 11s
2020-11-20 18:05:26.000174: furcate.runner] Thread 4 finished - 02m 21s - est. total time remaining: 06m 56s
2020-11-20 18:05:26.000174: furcate.runner] Thread 5 finished - 02m 22s - est. total time remaining: 06m 58s
2020-11-20 18:07:26.000297: furcate.runner] Thread 6 finished - 01m 10s - est. total time remaining: 04m 19s
2020-11-20 18:07:26.000297: furcate.runner] Thread 7 finished - 01m 11s - est. total time remaining: 04m 04s
2020-11-20 18:09:26.000419: furcate.runner] Thread 8 finished - 01m 11s - est. total time remaining: 01m 56s
2020-11-20 18:09:26.000419: furcate.runner] Thread 9 finished - 01m 15s - est. total time remaining: 01m 52s
```
 While running each thread will generate a log folder within the default `logs/` directory with a naming scheme
 dependent on the changing configurations. 
```
logs/
run_data.csv
|-- data0_data_e20_mnCNN_lr0-0001/
    | -- accuracy.jpg
    | -- data0.err
    | -- data0.log 
    | -- history.json
    | -- model.h5
    | -- run_data.json
|-- data2_data_e20_mnCNN_lr0-01/
|-- data6_data_e10_mnCNN_lr0-0001/
|-- data10_data_e10_mnInceptionV3_lr0-001/
|-- data3_data_e20_mnInceptionV3_lr0-0001/
|-- data7_data_e10_mnCNN_lr0-001/
|-- data11_data_e10_mnInceptionV3_lr0-01/
|-- data4_data_e20_mnInceptionV3_lr0-001/
|-- data8_data_e10_mnCNN_lr0-01/
|-- data1_data_e20_mnCNN_lr0-001/
|-- data5_data_e20_mnInceptionV3_lr0-01/
|-- data9_data_e10_mnInceptionV3_lr0-0001/
```

Log output folder format is `{data_name}{thread_id}_{data_dir}_{..variables..}/`
