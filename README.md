This repository contains code, examples and diagrams in response to 
reviewers of the paper "Superposition Gaussian Cox Processes".

In ec_criterion_diagrams, we include code and diagrams exhibiting the
calculation of the EC criterion for different orders ($m$ in the paper) of our
model. The U-shaped curve shows that the reviewer's hypothesis that the EC
criterion will favour in general more "complex" models is not borne out. In
this we are assuming that by "complex" the reviewer means higher order models.

in basis_function_observation_noise, we include tables of KS-test results for
the three synthetic intensity examples in the paper, for each of 5 different
sets of basis functions. The reviewer's hypothesis that the distribution of the
orthogonal series coefficient will  vary wildly in different bases is, in our
opinion, not borne out, given that (as expected) only the initial basis
functions $\phi_0$ appear to fail the KS-test in general. The use of the Normal
prior is still valid, given the two moment conditions of Campbell's theorem and
the maximum entropy principle.

In mnist_classification, we include an example, with code and diagrams, of
using a very simple autoencoder for dimension reduction, followed by our
classifier method on the latent space. This exhibits the fact that our
classification model is capable of being applied to higher dimensional models
if mapped to appropriate latent spaces. probabilities according to the model.
Included in the folder are the encoder and decoder weights, as well as the code
that generated the weights and constructed the prediction examples.
