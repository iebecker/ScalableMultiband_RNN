# ScalableMultiband_RNN

This repository contains the code to execute the model presented in [Multiband embeddings of light curves](https://www.aanda.org/articles/aa/full_html/2025/02/aa47461-23).

To use the code, a `Pipfile` and `Pipfile.lock` have been provided to replicate the Python environment. 
To install dependencies using Pipenv, run:
```bash
cd path/to/ScalableMultiband_RNN
pip install pipenv  
pipenv install
```

The code is meant to be used as a Python library. To use the code and its structure, you have two options:

1. Add the `src` path to the `PYTHONPATH` environment variable using
```bash
export PYTHONPATH=path/to/ScalableMultiband_RNN/src/:$PYTHONPATH
```

2. Use the Python `sys` library to add the `src` folder to the path.
```python
import sys
sys.path.append('path/to/ScalableMultiband_RNN/src/')
```

Currently, it only provides support for Python 3.11, TensorFlow 2.15.1 and its compatible CUDA implementation. For more information, refer to [these tables](https://www.tensorflow.org/install/source#tested_build_configurations).