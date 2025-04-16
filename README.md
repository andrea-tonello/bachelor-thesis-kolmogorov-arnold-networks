# Evaluating Kolmogorov-Arnold Networks (KANs) for Classification and Regression Tasks

## Overview
This repository contains the implementation and analysis of Kolmogorov-Arnold Networks (KANs), a novel neural network architecture based on the Kolmogorov-Arnold Representation Theorem (KART). 
The project compares KANs with traditional Multi-Layer Perceptrons (MLPs) on image classification and regression tasks, evaluating their performance, efficiency, and interpretability.

## Key Features
- KAN Architecture: Implementation of KANs using B-splines for activation functions, allowing local manipulations and adaptive precision.
- Comparative Analysis: Performance comparison between KANs and MLPs on datasets like MNIST, FMNIST, CIFAR10, CIFAR100, and regression datasets.
- Control Grid Extension: Investigation into how the control grid size affects model performance and overfitting.
- Interpretability: Exploration of KANs' potential for symbolic regression and feature visualization.

## Results Highlights
**Image Classification**
- MNIST/FMNIST: KANs perform comparably to MLPs, with slight advantages in smaller models.
- CIFAR10/CIFAR100: KANs show higher accuracy but suffer from significant overfitting, especially in larger models.
- Training Time: KANs generally require more time to train, particularly in deeper architectures.

**Regression tasks**
- Performance: KANs outperform MLPs in most regression scenarios, especially with smaller models.
- Efficiency: Faster convergence observed in KANs, though training times remain longer than MLPs.
- Overfitting: Less pronounced in regression tasks compared to classification.
