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
At the second approach we try to simplify the functionality of the first API, the API will be able to handle Supervised Learning and GANs.
- If the user gives to the Model single model/optimizer/loss functions, then the API will treat this as Supervised Learning manually.
- If th user gives to the Model 2 models/optimizers/loss functions, then the API will treat this as GANs.
- If the user gives the Model anything else, that won't work.

## Third Approach
At third approach we mainly focus on making the API more generic to help the users extend it and implement their own `train_step` method.
- Supported DDP and AMP for single model/optimizer/loss function.
- Ignored implementing a `train_step` for GANs and let the users extend the API and implement it.
- The API now has methods to handle just one model, however in case the user wanted to train GANs, the API can still validate them and handle them.

## Fourth Approach 
At fourth approach we update the third one, but we keep the same logic to handle GANs and more complex `train_step` method.
But at this approach we use `dict` for multiple inputs (models, optimizers, loss_fns), as dicts will add more flexibility and useability.
- Supported DDP for single and multiple models/optimizers/loss functions.
- The API can train one model, however in case the user wanted to train GANs, the user just needs to override `train_step` and feed a dict of models/optimizers/loss_fns and just `fit` the model.

## Fifth Approach 
At fifth approach we update the previous one to see how it's going to perform in some example, and I added some examples too. And almost it's the same logic as the last one.
- Added simple supvervised learning example.
- Added GAN example.
- Added DDP example.
Everything seems to be working fine, but we are thinking about approach more intuitive, as most of the team don't like using `dicts`, we see it's too free.

## Sixth Approach
At sixth approach we choose a totally different approach to avoid the drawbacks in the previous approaches, we provided methods like `set_data()` , `set_distributed_config()`, etc. This way we avoided using dictionaries for taking the inputs.
We also changed our logic for training GANs, if the user wants to train GANs, then he must override the `__init__()` and `train_step()` methods, the API and `fit()` method will be able to handle the models/optimizers/loss functions without overriding anything else.
- Added examples for all use cases.
- Changed the logic and the design.
- Used a non-RAII technique for this API instead of RAII (Resource Acquisition Is Initialization).

**Updates**:
- Used `train_engine` and `val_engine` as attributes.
- Added 4 methods for handlers. (2 for training and 2 for validation).
- Handled validation and setting data for validation via adding bool arg `train` in `set_data` method.

## Seventh Approach
After our previous experimentations, we realized how hard it is to find the generic and powerful solution  we are looking for, we have decided to implement and API similar to an existing solution, then modify it and make it cover all the feautres we want.
So this approach is inspired from Argus API implementation with our touches.
- Used `train_engine` and `val_engine` as attributes.
- Covered DDP via a separated method `set_distributed_config`.
- Handlers and metrics are feeded to the API via `fit()` and `validate()` methods, not separated methods.
- The data is also gived to the API via `fit()` and `validate()`, not with `set_data()` methods.

## Chosen approach
We at PyTorch-Ignite didn't find the proper solution yet.

## Acknowledgement
Thanks to PyTorch-Ignite, Argus, and MONAI teams for their help in the process of building this API.

