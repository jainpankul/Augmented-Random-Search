## Augmented-Random-Search

The algorithm is proposed by Horia Mania,  Aurelia Guy and Benjamin Recht. For complete paper, please check https://arxiv.org/pdf/1803.07055.pdf

Augmented Random Search (ARS) is a random search method used for continuous control problems. It is done by augmenting the basic random search method with three simple features.

1) scale each update step by the standard deviation of the rewards collected for computing that update step
2) normalize the system’s states by online estimates of their mean and standard deviation
3) discard from the computation of the update steps the directions that yield the least improvement of the reward

### Installation

pip install gym==0.10.5

pip install pybullet==2.0.8

conda install -c conda-forge ffmpeg

These commands above only install what you need to get the PyBullet environments and display them on your monitor. No AI framework such as TensorFlow or PyTorch needs to be installed.

