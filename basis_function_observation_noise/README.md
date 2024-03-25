In this folder we store results from ks-tests against normality for the 
orthogonal series estimate coefficients of the basis functions for 5 different
sets of basis functions, with points generated from three different intensity functions.
The intensity functions $\lambda_i: \mathcal{X} \rightarrow \mathbb{R}$
are as noted in the paper.

The  Hermite, Laguerre, Chebyshev_1, and Chebyshev_2 basis functions are
written as 
$\phi(x) = c_i, P_i(x) w^{1/2}(x)$
where $P_i(x)$ is the $i$-th orthogonal polynomial with respect to the measure
$w(x)$, and $c_i$ is a normalising coefficient.

The constructed bases are:
Hermite: The Hermite basis functions that comprise the standard decomposition of the
square exponential kernel with respect to the Gaussian distribution (Fasshauer 2012).
Laguerre: The Laguerre basis functions that comprise the standard decomposition of the
Matern kernel with respect to the exponential distribution (Tronarp and Karvonen, 2022).
Chebyshev_1: Chebyshev basis functions, constructed using the Chebyshev polynomials of the first kind.
Chebyshev_2: Chebyshev basis functions, constructed using the Chebyshev polynomials of the second kind.
             These are the same basis functions as used in the paper. The
             numbers in this table will differ from the tables in the paper
             because we re-ran these experiments in response to the reviewers'
             comments.
Sin: Simple Sine basis functions, constructed using the sine functions.
