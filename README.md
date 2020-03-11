## Causal Python Experiments

Some experimental scripts about causal models & bayesian networks in Python.

### Causal Bayesian Networks

### Causal Discovery Toolbox

I used the lucas dataset to check the output of the different causal discovery tools and compare them vs a true causal model (can be explore in `lucas_true.dot`). The dot files with the corresponding weights for each algorithm can be found in the `models` file.

#### Running the code

You must have docker & docker-compose installed. I am using a docker image provisioned with all the R and Python packages needed for `CDT` to run without issue.

First:
`docker-compose build`

then:
`docker-compose run --rm cps python3 try_cdt.py`
