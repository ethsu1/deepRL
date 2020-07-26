# What is this page about?
I've always been interested in machine learning and reinforcement learning. So I wanted
to document this project experimenting with machine learning and reinforcement learning.
I wanted to note down any breakthroughs/insights in my understanding to reinforcement my own learning 
as well as help someone else's learning.


# How does one utilize machine learning models in reinforcement learning algorithms?
Something that always troubled me when trying to understand how machine learning
could be used in the reinforcement learning framework was that in machine learning, 
(at least for supervised learning), there is a label. A label defines a ground-truth output, meaning
for a data in training set, the input data had a corresponding output label (classification) 
or output value (regression). So it made sense of how you could training a machine learning model
on the data. But with reinforcement learning, there is no associated ground-truth label/value. 
I did not understand howw I could manipulate the reinforcement learning framework to satisfy 
the general training framework of machine learning. That's when I learned that the machine learning
model is trying to approximate the Q value table. In classical (strictly) Q learning implementations,
you have a table/dictionary to map state-action pairs to Q-values. In deep Q learning, the machine learning
model is trying to approximate the underlying function of the Q-table. So the 'true values'
would be the temporal difference updates aka the new Q value at a particular state
(R(s) + discount * max_a Q(s',a)). The model will output the particular Q value for state and action Q(s,a)
and the 'true value' will be (R(s) + discount * max_a Q(s',a))



# What's the difference between SARSA and Q Learning?

# Why use deep Q Learning or Q learning with ML function approximators?
The goal of Q learning is to find the optimal policy by finding the optimal Q-values (aka the max expected future reward of state-action pairs).
In the basic case, the environment contains discrete states and discrete actions. Therefore, one can do Q learning
via values in a lookup table. But when there are continuous-valued states, the lookup table will be infinitely big.
This is where Q learning with some sort of function approximator comes in because we can feed in continuous-valued
states. Now our function approximator represents the lookup table in the basic case.

# When does Q Learning and SARSA fail?
Q learning (even with its ML variants) doesn't led well to real world scenarios because the action space isn't discrete.
You can no longer find the action that leads to max Q value because there infinite potential actions. For example,
one of the interesting applications of reinforcement learning is robotic control. Take a robotic hand with 5 fingers. For it learn
to pick up an object, there are infinite number of actions (the angle of each finger has infinite number of values, the force applied by each finger has infinite number of values, etc.). So Q learning wouldn't work well in these scenarios. You might be able
to discretize the actions but it probably wouldn't work that well or solve whatever task you are trying to solve

# What to do for continuous states and continuous actions?


# ML Models Used
I wanted to experiment with using different machine learning models to estimate the Q function. The complexity of the models
ranged from something as simple as linear regression to something more powerful like a neural network. 
It seemed very likely that the environment could not be solved with a linear model, considering when I used
a linear regression model, the agent was training on more episodes of the game and still producing very lackluster 
results. However, it seemed as if the agent was able to learn using linear regression, but it wouldn't have 
created a result that "solved the environment" (LunarLander: have a score of over 200 points over the last 100 episodes). The states of the environment
and respective "good" actions don't necessarily have a linear relationship with the Q-value, which makes sense that Q learning with
a linear regression model would not work well to solve the environment compared to a more complicated model, 
like a neural network (which has nonlinear capabilities). For example, a larger value for a right velocity (state) and adding more
right boost (action) won't necessarily lead to higher expected future reward. After trying to model the Q function
with linear regression and a quasi SVM regression, the end results were not too promising. Watching the agent
in the environment after training of 2000 episodes, you can tell the agent learned something but obviously 
it did not learn the optimal polciy. On the other hand, the neural network was able to find an optimal policy reasonably 
well, considering it finished training in ~350 episodes. Now that I had my fun messing around with different ML models 
with reinforcement learning algorithms, I wanted to recreate models that would "solve" the environment.
![Alt text](http://images/Neural_Network_Q_Learning.png?raw=true "Title")

# Why and when to have experience replay?
Experience replay helps the model learn more quickly because now it is not learning strictly
from consecutive samples. The model will sample from its memory and use that sample as input into the
output Q function before calculating the loss between the output Q value and target Q value.
Experience replay will be used once the memory data structure has been filled to whatever batch size specified.
This is when the actual training part of the RL model will occur. So that means in my case,
the very first 64 steps the model will be following the epsilon greedy policy based on the randomized
weights of the machine learning model (neural network, decision tree, etc), so there is technically no training.
After the first 64 steps, there would be training (computing and minimizing loss) after every iteration.


# Why copy weights from the prediction model to the target model?
If we use the same network for target Q-values and prediction Q-values, target Q-values and prediction Q-values
will continue moving with each iterations. It makes it more difficult to get our prediction Q-values to
approximate the target Q-values if the target Q-values keep moving. Hence we have a separate model for
target Q-values and we copy the weights from our prediction model to our target model every so often. This
way the target Q-values are not moving (more stable).


