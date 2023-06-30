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

"""

part1_q2 = r"""
**Your answer:**

"""

part1_q3 = r"""
**Your answer:**

"""

part1_q4 = r"""
**Your answer:**


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


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**


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
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
