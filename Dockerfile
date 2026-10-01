# Build:
#   docker build -t nngp-project .
#
# Default run: MNIST and Fashion-MNIST, identical settings, prints results.
#   docker run nngp-project
#
# Save the PNGs to your machine by mounting a folder:
#   mkdir -p output
#   docker run -v "$(pwd)/output":/nngp/output nngp-project
#
# Choose datasets (e.g. CIFAR-10):
#   docker run nngp-project cifar10
#   docker run nngp-project mnist fashion_mnist cifar10

FROM --platform=linux/amd64 tensorflow/tensorflow:1.15.0-py3

WORKDIR /nngp

# matplotlib isn't in the base TensorFlow image, and is needed for the plot
RUN pip install --no-cache-dir matplotlib

# Pre-download datasets at build time
RUN python -c "import tensorflow as tf; tf.keras.datasets.mnist.load_data(); tf.keras.datasets.fashion_mnist.load_data()"

# Copy everything in the repo (original nngp source, grid_data/, and
# uncertainty_plot.py) into the image.
COPY . /nngp

# Two fixes needed for this old codebase to run on a current image, baked
# in at build time instead of by hand each container session:
#   1. Python 2 -> 3 (xrange doesn't exist in Python 3)
#   2. numpy now defaults to allow_pickle=False; the precomputed grid files
#      in grid_data/ were saved as pickled object arrays
RUN sed -i 's/\bxrange\b/range/g' *.py && \
    sed -i "s/np.load(f)/np.load(f, allow_pickle=True, encoding='latin1')/" \
        nngp.py && \
        # Make sed patches fail loudly
        ! grep -n "xrange" *.py && \
        grep -q "allow_pickle=True" nngp.py


# `docker run nngp-project` with no extra args reproduces the MNIST panel;
# any arguments passed to `docker run` after the image name replace the
# CMD list below and go straight to uncertainty_plot.py.
#ENTRYPOINT ["python", "uncertainty_plot.py"]
#CMD ["--dataset=mnist", "--num_train=1000", "--num_eval=1000", \
     #"--hparams=depth=3,weight_var=2.0,bias_var=0.2", \
     #"--nonlinearities=tanh,relu", \
     #"--output_file=/nngp/output/uncertainty_fig3_mnist.png"]


# create output directory for saving the plots
RUN mkdir -p /nngp/output
ENTRYPOINT ["bash","run_all.sh"]



