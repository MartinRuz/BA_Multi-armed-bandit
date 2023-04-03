the accompanying python-script simulates a multi-armed bandit with two different policies as documented in my thesis.
the index 100 in lines 5-9 marks the number of repetitions of the same experiment.
N is the number of rounds, the policy choosese N times.
d is the number of possible choices in each round.
in the for-loop 17-23 we fill the array m with random normal distributed values, with mean 100+r and variance v, where r is a random number between -1 and 1 and v is a random number between 0.05 and 0.25.
Then we perform the two algorithms and print the results.