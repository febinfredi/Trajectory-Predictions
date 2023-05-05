# Trajectory-Predictions

### *CS 541: Deep Learning - Worcester Polytechnic Institute, Spring 2023*
### Members: Febin Fredi, Nihal Suneel Navale, Puru Upadhyay

## Abstract
Predicting the trajectories of pedestrians is a difficult task since it depends on a number of external factors. The context of the scene and the interaction between pedestrians are two of the most important factors. Several methods have been explored in past to tackle this issue transitioning from physics-based models to data-driven models based on sequence based neural networks like RNNs. In this project we have compared the performance of Long-Short terms memory and Transformer Architectures. Since, attention is the most important aspect for trajectory prediction, spatial and temporal context of trajectory are taken into account for making future predictions in interaction aware LSTM and Transformer architecture.We use mean average
displacement and final average displacement as comparison metrics for architecture comparison.

## Setup/Run
Install the following dependencies when git cloning, there might be others that might not be mentioned here, be sure to install them.
```
  sudo apt-get install ros-noetic-amcl
```
```
  sudo apt-get install ros-noetic-gmapping
```
```
  sudo apt-get install ros-noetic-move-base
```
