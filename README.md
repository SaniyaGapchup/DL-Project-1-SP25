# Detailed Report on Modified ResNet for CIFAR-10 Classification with Squeeze-and-Excitation Blocks

## 1. Introduction

Image classification remains a cornerstone task in computer vision, and Convolutional Neural Networks (CNNs) have continuously evolved to meet increasing performance demands. Residual Networks (ResNets) have been especially impactful due to their ability to train very deep networks using skip connections that mitigate the vanishing gradient problem.

In this project, we design a modified ResNet for CIFAR-10 classification that:
- **Maximizes Accuracy:** Targets state-of-the-art performance on CIFAR-10.
- **Optimizes Efficiency:** Keeps the model under a strict budget of 5 million trainable parameters, making it ideal for deployment on edge devices and IoT applications.
- **Enhances Feature Representation:** Integrates Squeeze-and-Excitation (SE) blocks to recalibrate channel-wise features dynamically, boosting the network's representational power.
- **Improves Convergence:** Utilizes advanced training strategies such as data augmentation, a Cosine Annealing learning rate scheduler, and the Lookahead optimizer to ensure robust training.

Inspired by *Efficient ResNets: Residual Network Design* by Thakur, Chauhan, and Gupta ([arXiv:2306.12100](https://doi.org/10.48550/arXiv.2306.12100)), our approach leverages these modern techniques to build an efficient yet powerful network.

## 2. Methodology

Our methodology is divided into three main parts: model architecture & hyperparameters, optimization and training strategy, and implementation details.

### 2.1 Model Architecture & Hyperparameters

#### Modified ResNet Architecture

- **Residual Blocks:**  
  The network is composed of multiple stages, each containing a set of residual blocks. Our configuration uses three stages with block counts `[4, 4, 3]`. Each block consists of two convolutional layers with batch normalization and ReLU activations, along with identity shortcut connections to facilitate gradient flow.

- **Convolutional Layers:**  
  - **Main Convolutions:** Each block uses a 3×3 convolution (`conv_kernel_sizes: [3, 3, 3]`) to extract local spatial features.
  - **Shortcut Connections:** Identity mapping is achieved via 1×1 convolutions (`shortcut_kernel_sizes: [1, 1, 1]`) when the number of channels changes or when downsampling is needed.

- **Channel Depth and Pooling:**  
  The network begins with 64 channels (`num_channels: 64`) and concludes with an average pooling layer (`avg_pool_kernel_size: 8`) to reduce the spatial dimensions before classification.

#### Squeeze-and-Excitation (SE) Blocks

- **Purpose:**  
  SE blocks are integrated to recalibrate the feature maps by modeling channel interdependencies. This mechanism allows the network to emphasize informative features while suppressing less useful ones.

- **Implementation Details:**  
  - **Squeeze:** Global average pooling compresses the spatial dimensions, producing a channel descriptor.
  - **Excitation:** The descriptor passes through two fully connected (or 1×1 convolutional) layers with a ReLU activation followed by a sigmoid function, yielding channel-wise weights.
  - **Recalibration:** These weights multiply the original feature maps to dynamically adjust channel responses.

In our configuration, the SE block is enabled (`squeeze_and_excitation: True`), which provides a significant boost in discriminative capability with minimal extra parameters.

#### Hyperparameters Summary

- **Model Hyperparameters:**
  - `num_blocks`: [4, 4, 3]
  - `conv_kernel_sizes`: [3, 3, 3]
  - `shortcut_kernel_sizes`: [1, 1, 1]
  - `num_channels`: 64
  - `avg_pool_kernel_size`: 8
  - `drop`: 0.2 (Dropout rate applied within residual blocks)
  - `squeeze_and_excitation`: True

### 2.2 Optimization and Training Strategy

#### Data Preparation

- **Data Augmentation:**  
  Random cropping (with padding) and random horizontal flipping are applied to enhance the training set diversity (`data_augmentation: True`).

- **Normalization:**  
  Images are normalized using CIFAR-10’s mean and standard deviation (`data_normalize: True`).

#### Training Parameters

- **Batch Size & Data Loader:**  
  - Batch size: 128  
  - Number of workers: 8

- **Training Duration:**  
  The model is trained for 300 epochs (`max_epochs: 300`).

#### Optimization Techniques

- **Base Optimizer:**  
  Stochastic Gradient Descent (SGD) is used with:
  - Learning rate (`lr`): 0.1
  - Momentum: 0.9
  - Weight decay: 5e-4

- **Lookahead Optimizer:**  
  To stabilize and accelerate convergence, SGD is wrapped with a Lookahead optimizer:
  - Lookahead steps (`lookahead_k`): 5
  - Interpolation factor (`lookahead_alpha`): 0.5

- **Learning Rate Scheduler:**  
  A Cosine Annealing scheduler (`lr_sched: "CosineAnnealingLR"`) gradually reduces the learning rate over training epochs.

- **Gradient Clipping:**  
  Not applied in this configuration (`grad_clip: None`).

### 2.3 Implementation Details

- **Framework:**  
  The project is implemented in PyTorch, utilizing its flexible modules for neural network construction, data handling, and GPU acceleration.

- **Code Structure:**
  - **Model Definition:**  
    The ResNet is built with modular residual blocks that incorporate dropout and optional SE blocks. A helper function (`conv1x1`) is defined for 1×1 convolutions, crucial for shortcut connections and SE block operations.
  - **Training Pipeline:**  
    The training loop includes:
    - Forward pass with CrossEntropyLoss.
    - Backward propagation and gradient computation.
    - Optimizer updates with the Lookahead mechanism.
    - Logging of loss and accuracy metrics via TensorBoard.
    - Checkpointing when validation accuracy improves.
  - **Inference Pipeline:**  
    A custom dataset class loads test data from a pickle file (`cifar_test_nolabel.pkl`), and the model’s predictions are saved in a CSV file.
  - **Utility Functions:**  
    Additional scripts count the model parameters to ensure the total remains under the 5 million parameter threshold.

## 3. Experiments and Results

### Experimental Setup

- **Dataset:**  
  The CIFAR-10 dataset, containing 60,000 32×32 color images in 10 classes, is used for training and evaluation.

- **Training Process:**  
  The model is trained over 300 epochs with the above configurations. Data augmentation and normalization are applied to improve generalization, and the Cosine Annealing scheduler adjusts the learning rate smoothly during training.

### Results and Analysis

- **Training Convergence:**  
  The training and validation loss curves demonstrate a steady decline, while accuracy improves consistently. The integration of dropout and SE blocks aids in faster convergence and improved performance on the validation set.

- **Performance Metrics:**
  - **Training Accuracy:**  
    High training accuracy is achieved as the network effectively learns to classify CIFAR-10 images.
  - **Validation Accuracy:**  
    The model attains competitive validation accuracy, confirming its generalization capability.
  - **Inference:**  
    The custom inference pipeline processes test data and outputs predictions in a CSV file. Confusion matrices and classification reports further validate robust performance across classes.

- **Model Efficiency:**  
  A parameter counting utility confirms that the model's trainable parameters remain under the 5 million limit. The SE blocks contribute significant performance improvements with minimal parameter overhead.

- **Comparative Insights:**  
  Despite the reduced parameter count, the modified ResNet shows competitive performance compared to larger models like ResNet18, demonstrating that careful architectural design and advanced optimization techniques can yield high accuracy with fewer resources.

## 4. System Specifications

The experiments and training were conducted on a high-performance GPU cloud instance with the following specifications:

- **Environment:** GPU Cloud Instance
- **CPU:** 32 vCPU
- **GPU:** Nvidia RTC 4090
- **System Memory:** 120 GB
- **Python Version:** 3.10.2
- **CUDA Version:** v12.1
- **Torch Version:** 2.2.4

These specifications ensured efficient training and inference, allowing for rapid experimentation and robust performance evaluation.

## 5. Conclusion

This project presents a comprehensive exploration of a modified ResNet architecture for CIFAR-10 classification that strikes an effective balance between accuracy and efficiency. Key findings include:

- **Efficient Architecture:**  
  The model is designed to operate under a strict parameter budget (<5 million parameters) by employing a tailored residual block design with configurable kernels, dropout, and SE blocks.

- **Enhanced Feature Recalibration:**  
  The integration of Squeeze-and-Excitation blocks significantly improves the network’s ability to capture and prioritize informative features, boosting overall performance with minimal additional parameters.

- **Robust Training Strategy:**  
  Advanced techniques such as data augmentation, Cosine Annealing learning rate scheduling, and the Lookahead optimizer contribute to stable convergence and high accuracy.

- **Practical Applicability:**  
  The model is well-suited for deployment on resource-constrained devices, as demonstrated by the successful implementation and evaluation on a GPU cloud instance with high-end hardware.

Future work could include exploring further regularization methods, extending the architecture to additional datasets, and fine-tuning hyperparameters to push the performance envelope even further.

