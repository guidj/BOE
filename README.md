# Syndicato

Simulating context-free bandits.


## Reported Metrics

There are four reported metrics, which are saved at each snapshot:
  
  - Probability of selection per arm
  - Average reward per arm
  - Cumulative reward per arm
  - Global cumulative reward


### Average reward per arm: Upper Confidence Bound

There is an option of reporting the Upper Confidence Bound (UCB) along with the average reward per arm.
The parameter report-ucb defines this behavior. 

When UCB is reported, there is an option to discount the value, linearly.
This is strictly for plotting purposes. UCB is relative measure, and as such, should be used
to estimate the confidence of knowledge of one arm over other arms. Not as an absolute measure.  

The computed UCB at each snapshot is stored as is.

## Execution

TBD, but use with python3.