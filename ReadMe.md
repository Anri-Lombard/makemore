# Makemore
Makemore simply makes more of what it is given. It uses multiple character-level language models that predicts the next letter in a sequence of letters.<br />
<br />
interestingly one of the models is a transformer similar to the one used in GPT-3, which is kind of a big deal.

## Review of video and project
* In progress...

# Buidling Makemoore
## Part 1: The spelled-out intro to language modeling
### Part 1.1: Bigram approach
* Bigrams are a sequence of two letters. For example, the bigrams in the word "Makemore" are "Ma", "ak", "ke", "em", "mo", "or", "re". They are used to count the frequency of a letter following another letter. For example, the bigram "ak" appears 1 time in the word "Makemore", which is then used to calculate the probability of the next letter by the language model.
* We use the PyTorch library to build the bigram model. 
  * Pytorch is a deep learning library that is used to build neural networks. It is similar to Tensorflow, but it is more intuitive and easier to use.
  * We end up with the occurences of each bigram in the dataset, which is then used to calculate the probability of the next letter by the language model.
* We then sample from the distribution created (by torch.multinomial) to generate the next letter.
* Funny enough, we end up with terrible predictions. This is because the bigram model is not good at predicting the next letter in a sequence of letters. It is good at predicting the next word in a sequence of words, but not the next letter in a sequence of letters.
  * To convince ourselves that we are not crazy, and that the bigram model actually just works badly, we sample from a uniform distribution to get even more terrible results.
* Being good at tensor manipulation is crucual since some complicated operations are needed to build models.
  * One of these operations is broadcasting. Broadcasting is a way to perform operations on tensors of different shapes. For example, we can add a scalar to a vector, or a vector to a matrix, or a matrix to a tensor. This is done by broadcasting the smaller tensor to the shape of the larger tensor.
    * Andrej recommends to take a tutorial on broadcasting to understand it well, since it is easy to run into bugs when using it. (Very easy to create subtle bugs when using broadcasting)
* We created a model, but want to somehow evaluate the quality of a model.
  * We assign a probability to each bigram to scope it's quality. If the model is great each probability should be near 1, and if the model is bad each probability should be near 0.
  * Another measure is to use the Maximum Likelihood Estimate (MLE). MLE is the probability of the most likely sequence of letters. For example, if the most likely sequence of letters is "Makemore", then the MLE is 1. If the most likely sequence of letters is "Makemoree", then the MLE is 0.
    * Statisticians usually work with the log-likelihood, which is the log of the MLE. This is because the log-likelihood is easier to work with, and it is also easier to compare the log-likelihood of different models.
    * When the model is good log-likelihood is close to 0, and if it is bad log-likelihood is very negative. We want to work with loss functions, which has the convention of being low at 0 and high at infinity. So we negate the log-likelihood to get a loss function.
    * Goal: maximize likelihood, which is equavalent to minimizing the loss function (average of negative log-likelihood).
* Our model could give an infinity if a word is very unlikely. To prevent this we do model smoothing by adding 1 element-wise to the model.

### Part 1.2: Neural Network approach
* Since we now have a single-number loss function, we can use gradient descent to optimize the model.
* We create a bigram dataset for the model.
* To encode integers we use one-hot-encoding. For example, if we have 5 letters, then we encode each letter as a vector of length 5. The vector is all zeros except for a 1 in the index of the letter. For example, if the letter is "a", then the vector is [1, 0, 0, 0, 0]. If the letter is "e", then the vector is [0, 0, 0, 0, 1].
  * Pytorch has a function to do this called torch.nn.functional.one_hot.
* In progress...

# References
* [Andrej Karpathy's Makemore project](https://github.com/karpathy/makemore)
* [Makemore Part 1](https://www.youtube.com/watch?v=PaCmpygFfXo&ab_channel=AndrejKarpathy)
* [Makemore Part 2](https://www.youtube.com/watch?v=TCH_1BHY58I&t=76s&ab_channel=AndrejKarpathy)
* [Makemore Part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=30s&ab_channel=AndrejKarpathy)
* [Makemore Part 4](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=2185s&ab_channel=AndrejKarpathy)