from ldm.utils.maths import binomial_bayes_factor_one_sided

print(
    binomial_bayes_factor_one_sided(n=1009, k=540, p0=0.5, alternative_hypothesis=">")#, a=1, b=1)
)
