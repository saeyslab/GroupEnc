
# GroupEnc

**Encoder with group loss for global structure preservation**

---

David Novak (1,2), Sofie Van Gassen (1,2), Yvan Saeys (1,2)

(1) Department of Applied Mathematics, Computer Science and Statistics, Ghent University, Belgium

(2) Data Mining and Modeling for Biomedicine, Center for Inflammation Research, VIB-UGent, Belgium

This work has been accepted for presentation at BNAIC/BeNeLearn 2023.

---

Recent advances in dimensionality reduction have achieved more accurate lower-dimensional embeddings of high-dimensional data.
In addition to visualisation purposes, these embeddings can be used for downstream processing, including batch effect normalisation, clustering, community detection or trajectory inference.
We use the notion of structure preservation at both local and global levels to create a deep learning model, based on a variational autoencoder (VAE) and the stochastic quartet loss from the [*SQuadMDS*](https://github.com/PierreLambert3/SQuaD-MDS-and-FItSNE-hybrid) algorithm.
Our encoder model, called *GroupEnc*, uses a ‘group loss’ function to create embeddings with less global structure distortion than VAEs do, while keeping the model parametric and the architecture flexible.
We validate our approach using publicly available biological single-cell transcriptomic datasets, employing R<sub>NX</sub> curves for evaluation.

---

## Installation

GroupEnc is a Python package built on top of TensorFlow.
We recommend creating a new Anaconda environment for GroupEnc.

On Linux or macOS, use the command line for installation.
On Windows, use Anaconda Prompt.

```
conda create --name GroupEnc python=3.9 \
    numpy pandas
```

Next, activate the new environment and install `tensorflow`.
TensorFlow installation is platform-specific.
GPU acceleration, when available, is highly recommended.

### macOS (Metal)

```
conda activate GroupEnc
pip install tensorflow=2.9.2
pip install tensorflow-macos
pip install tensorflow-metal
```

Consult [this tutorial](https://developer.apple.com/metal/tensorflow-plugin/) in case of problems.

### Windows (CUDA)

```
conda install conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow<2.11"
```

Consult [this tutorial](https://www.tensorflow.org/install/pip#windows-native) in case of problems.

### Linux (CUDA)

```
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
```

In a new terminal session, run:

```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Consult [this tutorial](https://www.tensorflow.org/install/pip#linux) in case of problems.

### CPU

```
pip install tensorflow
```

Consult [this tutorial](https://www.tensorflow.org/install/pip#cpu) in case of problems.

### TensorFlow Verification

To verify correct installation of TensorFlow, activate the environment and run the following line:

```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

This should return a non-empty list.


### R<sub>NX</sub> curves

Unsupervised evaluation of dimensionality reduction can be done using R<sub>NX</sub> curves.
This may require a large amount of RAM an may be intractable on large datasets or less-performant machines.

To compute R<sub>NX</sub> curves, install the `nxcurve` package:

```
pip install nxcurve
```

## Usage

To create an embedding using a VAE or GroupEnc model, use the `./scripts/embed.py` script with specified arguments.
To evaluate an existing embedding, use the `./scripts/score.py` scripts with specified arguments.

To run a benchmark on an HPC, use `./scripts/benchmark_embed.py` and `./scripts/benchmark_score.py` as starting points.

All scripts are documented, use the `-h` or `--help` flag to see usage.

