# RNN Character Level Language Model

[![license](https://img.shields.io/packagist/l/doctrine/orm.svg)](https://github.com/samre12/deep-trading-agent/blob/master/LICENSE)
[![dep1](https://img.shields.io/badge/implementation-tensorflow-orange.svg)](https://www.tensorflow.org/)
[![dep2](https://img.shields.io/badge/python-2.7-red.svg)](https://www.python.org/download/releases/2.7/)
[![dep3](https://img.shields.io/badge/status-in%20progress-green.svg)](https://github.com/samre12/deep-trading-agent/)\
Language Model based on [*The Unreasonable Effectiveness of Recurrent Neural Networks*](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) from [Andrej Karapathy's blog](https://karpathy.github.io).\
General Tensorflow implementation of LSTM based Character Level Language Model to model the probability distribution of the next character in the sequence given a sequence of previous characters.\
![charrnn](https://karpathy.github.io/assets/rnn/charseq.jpeg)
*The above image is taken from the mentioned blog.*\
For the complete details of the dataset, preprocessing, network architecture and implementation, refer to this [Wiki](https://github.com/samre12/charrnn/wiki).

## Requirements

- Python 2.7
- [Tensorflow](https://www.tensorflow.org/)
- [tqdm](https://pypi.python.org/pypi/tqdm) (for displaying progress of training)

## What's Interesting

This implementation will,

- provide support for arbitrary length input sequences by training the Recurent Network using **Truncated Backpropagation Through Time (TBPTT)**. It reduces the problem of vanishing gradients for very long input sequences.

- provide support for stacked LSTM layers with residual connections for efficient training of the network.

- provide support for introducing different types of *random mutations in the input sequence for simulating real world data like,
    1. dropping characters in the input sequence
    2. introducing additional white spaces between two words

- the input pipeline is based on Tensorflow primitive readers and queuerunners which prefetch the data making training upto 1.5-2X faster on hardware accelarators. Prefetching data reduces the total stall time of the hardware accelarators thus making their efficient use.

*Random mutations in the input sequence improve the robustness of the trained model against real world data.

## Implementation

- `tf.train.SequenceExample` for storing and reading input sequence lengths of arbitrary length

- `tf.contrib.training.batch_sequences_with_states` for splitting and batching input sequences for **TBPTT** while maintaining the state of the recurrent network for each input example

- `tf.nn.dynamic_rnn` for dynamic unrolling of each input example upto its actual length and not for the padding at the end. This is more correctness than for efficiency