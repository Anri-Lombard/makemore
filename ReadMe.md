# Makemore
Makemore simply makes more of what it is given. It uses multiple character-level language models that predicts the next letter in a sequence of letters.<br />
<br />
interestingly one of the models is a transformer similar to the one used in GPT-3, which is kind of a big deal.

# Buidling Makemoore
## Bigrams
* Bigrams are a sequence of two letters. For example, the bigrams in the word "Makemore" are "Ma", "ak", "ke", "em", "mo", "or", "re". They are used to count the frequency of a letter following another letter. For example, the bigram "ak" appears 1 time in the word "Makemore", which is then used to calculate the probability of the next letter by the language model.
* In progress..


# References
* [Andrej Karpathy's Makemore project](https://github.com/karpathy/makemore)
* [Makemore Part 1](https://www.youtube.com/watch?v=PaCmpygFfXo&ab_channel=AndrejKarpathy)
* [Makemore Part 2](https://www.youtube.com/watch?v=TCH_1BHY58I&t=76s&ab_channel=AndrejKarpathy)
* [Makemore Part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=30s&ab_channel=AndrejKarpathy)
* [Makemore Part 4](https://www.youtube.com/watch?v=q8SA3rM6ckI&t=2185s&ab_channel=AndrejKarpathy)