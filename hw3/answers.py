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
**Your answer:**
The first one is practical – The entire text is very large an loading it in its entirety onto the GPU memory is might
not be possible or is slow if possible. Similarly, the computational graph at each loss / gradients calculation and 
step would be very large.

The second reason is that with very large text the gradients would vanish/explode between characters with big gaps,
making the optimization task much harder.

Another reason is because of all the usual reasons we split the data in SGD – We wish to calculate the loss a lot
of times each epoch, updating the model at each step. This allows faster convergence, a higher chance of reaching
good minimums etc.
"""

part1_q2 = r"""
**Your answer:**
This is due to the structure of the model – the model learns weight which based on the input change what information 
is passed across the sequence length, and so, if some context is not changed the model wouldn’t change it, no matter
how long the sequence it is fed.

For example in our task, if the model has some values in $h$ corresponding to which character is talking, and this 
context won’t change until a name with capital letters would be introduced followed by “:” the model might not forget
those values of $h$ for as long as it sees (or predicts) outputs of stage directions represented by capital letters, 
even if the sequence length had passed.
"""

part1_q3 = r"""
**Your answer:**
We are not shuffling the order of batch when training because the order of them is important for the hidden state 
variable the model keeps. Even if we don’t propagate loss through batches the model is still better trained when being 
fed his hidden state on the last character of the last batch.

The alternative (shuffling) is to make the model’s learning task simply harder by feeding it unrelated hidden 
state at the beginning of batch. This would also hurt the long-term memory of the model since it would be incentivized 
to forget the context quickly between batches.
"""

part1_q4 = r"""
**Your answer:**
1.   We lower the temperature for sampling to a value below 1 because we wish to a more distinct loss for our model.
Meaning that for prediction with the right character having the biggest score we wish to get low losses, and for
prediction with the right character having a smaller score we won’t the loss to be larger.

      This is especially important in character predictions, where a lot of options might be decent and so we wish 
      to have sharper losses specifically when the model is uncertain about two characters, for example “h” or “o” 
      after the letters “ t”, both options are good but only one is the right prediction in the current context and 
      we wish our model to learn it – changing a lot when the model is wrong, even if by a slight amount and not 
      caring that much if the model is correct even if by a large amount.

2.   When the temperature is very high, we get uniform distribution of the after-softmax probabilities, regardless 
of the scores, making the model nearly impossible to optimize, because the would be nearly constant at any score
values, making it’s gradient nearly 0.

3.    When the temperature is very low, we get “onehot” of the argmax score as the after-softmax probabilities, 
no matter what the scores are. Again in a small area around the current model parameters the probability and therefor 
the loss are nearly constants, making the gradient nearly 0 and the model nearly impossible to optimize.

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
**Your answer:**
The $\sigma^2$ hyperparameter influence on the weights we give to the data loss part of out loss. which is one 
of the 2 parts in our loss function:

1. The data loss - how close is the reconstructed image to the origin

2. The kldiv loss - how similar is the distribution of the latent space to the target distribution (in out case- 
standard normal distribution)

This hyperparameter directly influene the behavior of the model at the end of the training process.

Setting lower values of sigma will make the model "less care" about how the reconstrufted image is close to the images
it trained on, which leads to creating new images that are quite different from the model's dataset.

On the other hand, setting higher values of sigma will make model to care alot about making the reconstructed images
similar to the images in its dataset, which will make the final model creating images that are close to the images in the 
dataset. in such model we will not except to get special images that we haven't seen like them before.


"""

part2_q2 = r"""
**Your answer:**

1. The reconstruction part in the loss function represents how well our decoder can make an image that
looks like the original input. the porpuse of this part in the loss function is to help the model in generating 
new images that will bw similar to the input images during the trainig process

The KLdiv part of the loss represents how similar is the learned distribution of the latent space to our target 
distribution (which is standard normal distribution). the purpose of the KLdiv's part is to help the model to generate
better samples from the latent space. (to make the distribution of the samples to be similar to the distribution of the 
data. as we called the "posterior distribution")

2. During training, the encoder network learns to map input data to the parameters of the latent distribution, which we 
consider as the mean and variance. The KL loss term compares this learned distribution to our target distribution
and calculates the dissimilarity between them. This means that the KLdiv loss influence the learning process of the latent
space, because it affects its differentiation, and tries to make it closer to a certain distribution.

3. The first significant benefit of the KLdiv part is that it constitutes as a regularization term which prevent
    overfitting and improves the generalization.
    Another benefit of this term is that it makes for smooth and continuous latent space. This is reached by making
    the distribution of the latent space closer to the normal standard distribution.

"""

part2_q3 = r"""
**Your answer:**

The evidence distribution represents the likelihood of reconstructing an image that will resemble the input image.
By maximizing this term, we encourage the model to reconstruct images that will be as faithfully as possible to the 
original data, which will eventually coase for a more accurate model.



"""

part2_q4 = r"""
**Your answer:**

we model the **log** of the latent-space variance corresponding to an input, $\bb{\sigma}^2_{\bb{\alpha}}$, in
order to achieve the following purposes

1. Stability and numerical precision: Variance values can vary over a wide range, and directly modeling them may lead to numerical instability.
Taking the logarithm of the variance helps ensure numerical stability and precision during the training process. 

2. *************this is what the chat says:*********************************8
Constraint on Positive Values: Variance values must be positive since they represent the spread or uncertainty in the latent space. However, when 
directly modeling the variance, there is no inherent constraint to ensure positive values. By modeling the logarithm of the variance, we implicitly 
enforce the constraint that the variance itself must be positive, as exponentiating the logarithm will always yield a positive value.



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
**Your answer:**

Each encoder layer allows for represenetation of a certain token only in relation to the tokens that are
in $sliding-window-size * 0.5$ distance from it. That means that if we stack multiple encoder layers on top of each other, 
an arbitrary token "x" has context of farther tokens, because after the first layer the
neighbors tokens of "x" have context of their neighbors, that are not reachable from "x" directly
(because their distance from "x" is bigger than the sliding window size), and this context would
keep increasing like so at each layer.

Generally we can state that for $L$ layers each token representation has information from $1 + L\cdot w$ other tokens.

For example, assume the sliding window size is 4, and we have a series of 20 tokens.
After performing 1 encoder layer, the token in index 0 has context of the tokens in 
index smaller/equal to 3. And 3 has context of those in indices between 1 to 5.
So after the second encoder layer, the representation of the token at index 0 has context of the representation of the 
token at index 3, that already encodes the information of token at index 5. So, we can say that now the representation of the
token at index 0 has context of the token at index 5. 
Performing this process multiple times results in a broader context in the final layer.

"""

part3_q2 = r"""
**Your answer:**
We propose the “random and sliding windowed attention” which works as follows:

Each time you calculate attention matrix $A\in \mathbb{R} ^{n_X n}$ , calculate it at the $w + 1$ main diagonals
(like we implemented) and also random $n \cdot w$ indices in $A$ and calculate them.

This would propagate information throughout the sequence globally. Like in the previous question, after $L$ layers
each representation would have for certain context of $L \cdot w$ tokens, but for each token outside it’s certain
context, the probability of the token not being inside the context decreases exponentialy in L, the probability being
$(1-\frac{w}{n})^{context_{L-1}}$ where $ context_{L-1}$ is the amount of representations which token is in their context 
at last layer.


"""

part4_q1 = r"""
*Your answer:*
BERT achieves better results than out previeous transforner, (in around 15% better accuracy).
this is because BERT is a much larger model, trained on larger dataset. 
***********FINISH AFTER IMPLEMENTING PART ***********

these results wouldn't be the same for any "downstream" task, because in some tasks, BERT vast understanding
might not be as usefull as specifically built model. For example, BERT isn't very good in understanding what
something is not, as explained in this [article](https://towardsdatascience.com/bert-is-not-good-at-7b1ca64818c5)

another "downstream" task that BERT might not be usefull, is when reviews in hebrew are fanatically written in english
alphabet. in this case, BERT will perform much worst, because it relies on its english understanding. While the
model that is trained from scratch would have performed similarily. note: embedding might effect this scenario

"""

part4_q2 = r"""
**Your answer:**
In BERT model, the last linear layers are typically the classification layers. as such, we can relate to it as more 
task specific layers. This is unlike the internal layers, such as the multy headed attention blocks, that are rather
general language understanding layers than task specific. 

Considering this, if we fine-tune the internal layers instead of the two last linear layers, the model could perform
well, but it will be more challenging, as it would need to understand how good the review is depend on general ungerstanding
of the review. (such as its grammar - if it's written in first/second person for example). so eventually we can rather 
determine that this manner won't achieve better results.

to conclude, fine tuning the internal layers of the model would have been less efficient than fine tuning its last linear layers,
as we wish to train the model to perform well on a specific task and are not willing to change its general understanding
of the language.


"""


# ==============
