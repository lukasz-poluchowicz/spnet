# spnet

This is a capsule network implementation I developed to experiment with unsupervised learning in MNIST digits classification. The implementation is inspired by [Matrix capsules with EM routing](https://openreview.net/pdf?id=HJWLfGWRb) paper authored by Geoffrey Hinton, Sara Sabour and Nicholas Frosst.

The model is constructed like a standard convolutional neural network with the difference that a layer doesn't have one-dimensional channels but multidimensional capsule channels. Each capsule has a vector of features and an activation. Each layer of capsule channels has a decoder which tries to decode layer's output back into layer's input. Capsule layers are trained separately one by one from the lowest-level layer to the highest-level layer. Higher-level layer is trained only after training of all lower-level layers is finished. A layer is trained together with its decoder as an autoencoder.

I used a subset of two digits from MNIST dataset to train a small random model with four layers:

&#xfeff; | Number of capsule channels | Number of features per capsule | Kernel size | Stride
-------- |:--------------------------:|:------------------------------:|:-----------:|:------:
Layer 1 | 4 | 4 | 4 | 1
Layer 2 | 8 | 4 | 3 | 2
Layer 3 | 8 | 6 | 3 | 2
Layer 4 | 2 | 8 | 4 | 2    

The number of capsules in the last layer matches the number of classes in the training dataset (subset of two digits from MNIST dataset). The model is trained without labels, so after the training is finished, capsules in the last layer need to be assigned to the classes. There are two possibilities of such assignment and the one with the highest accuracy is chosen. This model trained without labels reached 79% test accuracy.

I tried to train another model on a subset of four digits from MNIST dataset, but I didn't get any meaningful result in reasonable time.
