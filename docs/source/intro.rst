Introduction
============

``furcate`` is a lightweight wrapper for automatically forking deep learning sessions to enable parallel model training across multiple GPUs. 

The aim is to provide a single object which users can subclass to perform various deep learning training automatically, taking away all of the hassle of data management and hyperparameter tuning. 

The current implementation has been developed in Python 3 and tested using Tensorflow. 

Motivation
**********

During my transition from software engineering into more machine/deep learning domain I found it tedious to just set up and manage the environment before I could even begin training a model. 

This package is intended to provide a quick, as well as a (hopefully) easy to understand, way of getting an environment set up to allow people to focus on just designing their models. 
