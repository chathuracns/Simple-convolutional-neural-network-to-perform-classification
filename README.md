# EN3150 Assignment 03: Simple Convolutional Neural Network for Classification

**Author:** Sampath K. Perera  
**Date:** October 1, 2024

## 1. CNN for Image Classification

In this assignment, we build a simple image classifier using a Convolutional Neural Network (CNN). You are free to use any programming language and toolkit, but **Python**, **TensorFlow**, **Keras**, or **PyTorch** are recommended.

### Steps:

1. **Set up your environment**: Ensure all required packages are installed.
2. **Prepare your dataset**: Choose a dataset from the UCI Machine Learning Repository (do not use CIFAR-10) and split it into 60% training, 20% validation, and 20% testing subsets.
3. **Build the CNN model**:
   - Convolutional layers with activation functions
   - MaxPooling layers
   - Flatten and fully connected layer
   - Dropout and output layer (Softmax activation)
4. **Train the model**: Train for 20 epochs, plot the training and validation losses, and use `adam` optimizer with sparse categorical crossentropy as the loss function.
5. **Evaluate the model**: Record train/test accuracy, confusion matrix, precision, and recall.
6. **Experiment with learning rates**: Test with different learning rates (0.0001, 0.001, 0.01, 0.1) and justify your final selection.

### Model Architecture:

- A Convolutional layer with x1 filters and m1×m1 kernel
- A MaxPooling layer
- Another Convolutional layer with x2 filters and m2×m2 kernel
- Another MaxPooling layer
- Flattened output
- A fully connected layer with x3 units
- Dropout with rate d
- Output layer with K units and softmax activation

## 2. Comparison with State-of-the-Art Models

Pretraining a CNN on a large dataset (e.g., ImageNet) and using it for transfer learning is common. In this assignment:

1. Choose two pre-trained models (e.g., ResNet, Googlenet, AlexNet, DenseNet, VGG).
2. Fine-tune the models on the same dataset as your custom CNN.
3. Compare the accuracy and performance of your model with these state-of-the-art models.

## 3. Additional Resources

- [MIT: Convolutional Neural Networks](#)
- [MIT: Introduction to Deep Learning](#)
- [PyTorch Training a Classifier](#)
- [PyTorch ResNet](#)
- [Keras Image Classification from Scratch](#)

## 4. GitHub Profile

Ensure that you commit regularly to your GitHub (or SVN) profile, as I will be checking the repository for continuous progress.

## 5. Submission

Submit the assignment as a ZIP file: **EN3150_groupno_A03.zip**. Include the report and code with the names and index numbers of your group members. Late submissions will receive a 15% penalty, and plagiarism will be penalized by 50%.

---

**References**:  
1. Kevin P Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022.  
2. Kunihiko Fukushima, *Cognitron: A Self-Organizing Multilayered Neural Network*, 1975.  
3. Yann LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, 1998.  

