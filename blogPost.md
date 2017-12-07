# Blog Post

There have been artists who try to copy others. In art class in elementary school, you might have tried to draw a picture in the style of
Van Gogh or someone like that. The art of transforming an existing image into another author's style is an idea that has already been
explored using neural networks. However, this problem is an interesting one when it comes to the realm of Natural Language Processing (NLP).
How do we rate how close author's styles are? What about the target author's text do we want to capture so that we can transform input text to
match the target's style? These questions are the sort of things that we explore in our project: Author Style Transfer Using Recurrent
Neural Networks.

## Prior Work

First, though, let's take a look at what has already been done. In "Applying Artistic Style Transfer to Natural Language," Edirisoorya and
Tenney make the point that using style transfer in the realm of NLP is a relatively new technique. Here, literary text was replaced with
embedding ID's and fed into a GRU for identifying the author. The network for the style transfer had an encoder and decoder that would enable
them to define a loss in content and style. Before we go on, we need to define their content loss and style loss.

Their Seq2Seq model pooled (averaged) consecutive word vector inputs along with word vector outputs, found the difference between these
vectors, and labeled the result as content loss. For style loss, it was a little more complex. Suppose an image was fed into the network. Then,
at every hidden layer, a vector will be generated. Now suppose we input the text whose style we want to transform. We can compare the vectors
that it generates at every layer and the vectors from earlier, and create a style loss function from there. Thus, we get a cost function that
is a linear combination of content loss and style loss and then create a network that minimizes this function.

The project did not work too well, and there were a few possible improvements that the paper mentioned. That said, we cannot come to any
scientific conclusion without trying the same thing at least a few times, right? Thus, we tried using this paper as our basis to attempt
to solve the problem of Author Style Transfer.

## The Data

So let's talk about where the data is coming from. Kaggle has a really nice dataset that it provided for the "Spooky Author Identification"
competition. The best part is that this data is clean and pre-processed very well, so we can focus on word embeddings using just this data.
However, data science is all about the data, so some code was written to try and process the data that came from Project Gutenberg, and that
taught us a few things about dealing with data.

Pre-processing takes a lot of effort, and it can many times come from the fact that data is inconsistent with each other. A text file for one
book can be radically different from the text file of another book. Also, people generally don't think about how a piece of software will read
data, so you can find random content in between paragraphs and even sentences in paragraphs can be hard to split, even if you use a library
like NLTK. Note for the future: Expect to spend considerable time just getting data ready.

Nonetheless, we also had the Kaggle Dataset, so we had 5 authors worth of data to work with. For the data from Project Gutenberg, we picked one
book each from Charles Dickens and Leo Tolstoy to process. With this massive dataset, we proceeded to train the model, which will be described
next.
