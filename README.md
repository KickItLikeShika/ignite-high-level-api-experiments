# ignite-high-level-api-experiments
A draft repository for experimenting different implementations and approaches for the new High-Level Model API at PyTorch-Ignite. (Part of GSoC'21).
The code of the experiments won't include much engineering aspects and not much implementation details, the aim of the code is just to introduce the approach as clear as possible.

Each approach is implemented in one file, and at the end of the file examples will be provided, mostly the examples will be about 2 parts:
- Simple use case (Supervised Learning).
- GANs.


## First Approach
At the first approach created a Model API that takes one or more models/optimizers/loss functions, the API is able to handle: 
- Supervised Learning (SL): By specify the `training_type` argument as "SL".
- GANs (GAN): By specifing the `training_type` argument as "GAN".
- Semi-Supervised Learning (SSL): By sepcifing the 'training_type' argument as "SSL". 

This approach is a bit complex and maybe confusing to the users, and we want to avoid this.
For now this is how the implementation looks like, maybe later we find a better way to polish this approach.


## Second Approach
At the secomd approach we try to simplify the functionality of the first API, the API will be able to handle Supervised Learning and GANs.
- If the user gives to the Model single model/optimizer/loss functions, then the API will treat this as Supervised Learning manually.
- If th user gives to the Model 2 models/optimizers/loss functions, then the API will treat this as GANs.
- If the user gives the Model anything else, that won't work.


## Chosen approach
We at PyTorch-Ignite didn't find the proper solution yet.

## Acknowledgement
Thanks to PyTorch-Ignite, Argus, and MONAI teams for their help in the process of building this API.

