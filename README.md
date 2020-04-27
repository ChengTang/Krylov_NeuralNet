# Krylov_NeuralNet
Training neural network using Krylov subspace method on a regression problem.

Unlike first-order optimizers, to adapt the optimizer for other problems, one needs to change the inference function and network structure 
due to the fact that evaluating the Hessian-vector product requires that one store all the trainable parameters in one giant vector (then slice and distribute them to appropriate 
layers when constructing the network).
