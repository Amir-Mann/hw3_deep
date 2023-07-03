r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Our answer:**

The first one is practical – The entire text is very large and loading it in its entirety onto the GPU memory might
not be possible or slow if possible. Similarly, the computational graph at each loss / gradients calculation and 
step would be very large.

The second reason is that with very large text the gradients would vanish/explode between characters with big gaps,
making the optimization task much harder.

Another reason is because of all the usual reasons we split the data in SGD – We wish to calculate the loss a lot
of times each epoch, updating the model at each step. This allows faster convergence, a higher chance of reaching
good minimums etc.
"""

part1_q2 = r"""
**Our answer:**

Firstly, the model is given context between batches in the form of the hidden state, and so the information is preserved
through longer text.

The model is also able to learn to preserve context outside of sequence length:
This is due to the structure of the model – the model learns weights which based on the input change what information 
is passed across the sequence length, and so, if some context is not changed the model wouldn’t change it, no matter
how long the sequence it is fed.s

For example in our task, if the model has some values in $h$ corresponding to which character is talking, and this 
context won’t change until a name with capital letters would be introduced followed by “:” the model might not forget
those values of $h$ for as long as it sees (or predicts) outputs of stage directions represented by capital letters, 
even if the sequence length had passed.
"""

part1_q3 = r"""
**Our answer:**

We are not shuffling the order of batches when training because the order of them is important for the hidden state 
variable the model keeps. Even if we don’t propagate loss through batches the model is still better trained when being 
fed his hidden state on the last character of the last batch.

The alternative (shuffling) is going to make the model’s learning task simply harder by feeding it unrelated hidden 
state at the beginning of batch. This would also hurt the long-term memory of the model since it would be incentivized 
to forget the context quickly between batches.
"""

part1_q4 = r"""
**Our answer:**

1. We lower the temperature for sampling to lower the variance of characters for each model prediction. When the temperature
is lower, the model has higher chance of sampling one of the characters that fit the context in its prediction. This
is good foe sampling because we dont want the model to randomly make "typos", we want it to actually generate text 
nearly deterministically.

2.   When the temperature is very high, we get uniform distribution of the after-softmax probabilities, regardless 
of the scores. This will cause the generated text to be nearly random.

3.    When the temperature is very low, we get “onehot” of the argmax score as the after-softmax probabilities, 
no matter what the scores are. This will introduce zero variance in the model prediction which might cause the model
to enter loops.

For example, before an actor exits, there is a large amount of spaces followed by the letter "E". 
We can see that the model generated random amount of spaces for such lines. We might expect that when the temperature is
extremely low, those line would be extremely long or even infinite.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
   
    # ========================
    return hypers


part2_q1 = r"""
**Our answer:**

The $\sigma^2$ hyperparameter influence on the weights we give to the data loss part of our loss.
The hyperparameter describes how much variance should the reconstructed images have around the original x.

Setting lower values of $\sigma^2$ will make model to care alot about making the reconstructed images
similar to the images in its dataset, which will make the final model creating images that are close to the images in the 
dataset. in such model we will not except to get special images that we haven't seen like them before.

Setting higher values of $\sigma^2$ will make the model "care less" about how the reconstrufted image is close to the images
it trained on, which leads to creating new images that are quite different from the model's dataset.
"""

part2_q2 = r"""
**Our answer:**


1. The reconstruction part in the loss function represents how well our decoder can make an image that
looks like the original input. the porpuse of this part in the loss function is to help the model in generating 
new images that will be similar to the input images during the trainig process

The KLdiv part of the loss represents how similar is the learned distribution of the latent space to our target 
distribution (which is standard normal distribution). the purpose of the KLdiv's part is to help the model to generate
better samples from the latent space. This will allow as to sample from $\mathcal{N}(\bb{0},\bb{I})$, decode and get
some reconstruction in the distribution $p(X)$ space.

2. The KL loss is a regularization for the learned mapping onto the latent space.
During training, the encoder network learns to map input data to the parameters of the latent distribution, which we 
consider as the mean and variance. The KL loss term compares this learned distribution parameter to our target distribution
and minimizes the dissimilarity between them.

3. The first significant benefit of the KLdiv part is that it constitutes as a regularization term which prevent
    overfitting and improves the generalization.
    Another benefit of this term is that it makes for smooth and continuous latent space. This is reached by making
    the distribution of the latent space closer to the normal standard distribution.
    Laslty it makes it easier to sample new unseen representations from the distribution.

"""

part2_q3 = r"""
**Our answer:**


The evidence distribution represents the probabilty of reconstructing an image that will resemble the input image.
By maximizing this term, we encourage the model to reconstruct images that will be as faithfull as possible to the 
original data, which will eventually cause for a more accurate model.
"""

part2_q4 = r"""
**Our answer:**


we model the **log** of the latent-space variance corresponding to an input, $\bb{\sigma}^2_{\bb{\alpha}}$, in
order to achieve the following purposes

1. Stability and numerical precision: Variance values can vary over a wide range, spesificly of small values, 
and directly modeling them may lead to numerical instability.
Taking the logarithm of the variance helps ensure numerical stability and precision during the training process. 

2. Variance must be positive: The variance must be positive, by modeling log variance we allow are model to output
value over $\mathbb{R}$, and then e to it's power gives as strictly positive variance values.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers



part3_q1 = r"""
*Our answer:*

Each encoder layer allows for represenetation of a certain token only in relation to the tokens that are
in $sliding-window-size * 0.5$ distance from it. That means that if we stack multiple encoder layers on top of each other, 
an arbitrary token "x" has context of farther tokens, because after the first layer the
neighbors tokens of "x" have context of their neighbors, that are not reachable from "x" directly
(because their distance from "x" is bigger than the sliding window size), and this context would
keep increasing like so at each layer.

Generally we can state that for $L$ layers each token representation has information from $1 + L\cdot w$ other tokens.

For example, assume the sliding window size is 4, and we have a series of 20 tokens.
After performing 1 encoder layer, the token at index 0 has context of the tokens in 
index smaller/equal to 3. And 3 has context of those in indices between 1 to 5.
So after the second encoder layer, the representation of the token at index 0 has context of the representation of the 
token at index 3, that already encodes the information of token at index 5. So, we can say that now the representation of the
token at index 0 has context of the token at index 5. 
Performing this process multiple times results in a broader context in the final layer.

"""

part3_q2 = r"""
*Our answer:*

We propose the “RASWA - random and sliding windowed attention” which works as follows:

Each time you calculate attention matrix $A\in \mathbb{R} ^{n \vectimes n}$ , calculate it at the $w + 1$ main diagonals
(like we implemented) and also random $n \cdot w$ indices in $A$ and calculate them as well.

This would propagate information throughout the sequence globally. Like in the previous question, after $L$ layers
each representation would have for certain context of $L \cdot w$ tokens, but for each token $t$ outside it’s certain
context, the probability of $t$ not being inside the context decreases exponentialy in $L$, the probability being
$(1-\frac{w}{n})^{context_{L-1}}$. Where $context_{L-1}$ is the amount of representations which $t$ is in their context 
at last layer.
"""

part4_q1 = r"""
Our answer:

BERT achieves better results than our previeous transforner, (around 15% better accuracy).
This happens because BERT is a much larger model, pre trained on larger dataset, and so has stronger understanding
of the english language then what can be obtained from our small dataset, with our smaller model which also only
uses windowed attention. 

These results wouldn't be the same for any "downstream" task (altough for most - they would be), 
because in some tasks, BERT's vast understanding might not be as usefull as specifically built model.
For example, BERT isn't very good in understanding what something is not, 
as explained in this [article](https://towardsdatascience.com/bert-is-not-good-at-7b1ca64818c5)

Another "downstream" task that BERT might not be great at, is when reviews in hebrew are fanatically written in english
alphabet (haseret haze haia mamash garua). in this case, BERT will perform much worst, because it relies on its english understanding.
While the model that is trained from scratch would have performed similarily. note: embedding might effect this scenario

"""

part4_q2 = r"""
*Our answer:*

The model would perform much worse while freezing the two last linea layers and fine-tuning only two internal layers.

In BERT model, the last linear layers are the classification layers. As such, they are generally more 
task specific layers. This is unlike the internal layers, such as the multy headed attention blocks, that are
general language understanding layers, and should be usfull for any task. 

Thus, fine-tuning two internal layers is a much harder task than doing so to the classification head. 
First of all, the layers we are optimizing are much farther from the output, making changes in them have expolding or
vanishing effect on the output. Secondly, the layers we are optimizing over have much less linguistic context, because they
are after less layers of self attension, making it harder for them to learn true lingual "goodness" or "badness" of a review.

"""

# ==============
