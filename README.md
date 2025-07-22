# MPWAE2

An autoencoder model that works on data from medical samples, compresses information into metapathway encoding, then reconstructs it as close as possible to the original input.

Made as a project for the Introduction to Data Mining class at University of Catania - Computer Science (L-31).

Originally, this project was made to work inside a [Google Colab Document](https://colab.research.google.com/drive/1s8aD9c0gD2y3k318R38TMNAjWPEDl9js?usp=sharing) which will be updated until project closure.

Therefore, some modifications may be needed, keep that in mind when working with code present in this repository.

## Project specifics

### Requisites

Overall structure:

- Input layer: $N$ node features
- ...
- Latent layer: $P$ values
- ...
- Output layer: $N$ node features

Loss Function: MSE

> **Personal log.**  Other loss functions are also valid, but this was requested by assignment.

### Files

These are the specifics of the files used to design and train the model architecture

> **Personal log.** Up to this point it isn't clear if the data provided can be uploaded in a public repository; only the file details will be shown until then.

```yaml
files:
  - file: metapathway_edges_simplified_2025.tsv
    notes: |
      "A TSV file with two columns containing
       the list of metapathway arcs;
       first column source node id,
       second column arc destination node id"
  - file: metapathway_nodes_to_pathways_2025.tsv
    notes: |
      "a TSV file that links each node to the
       related pathway to which it belongs;
       first column pathway id,
       second column pathway name,
       third column node id."
  - file: metapathway_nodes_2025.tsv
    notes: |
      "a TSV file listing the nodes contained in the metapathway;
       first column node id,
       second column node name."
  - file: test_tumor_samples.tsv
    notes: |
      "contains the sample data on which to do the training;
       within the file is stored an array where
       each column represents a sample and each row
       represents a node in the metapathway;
       each row has M+1 columns, because the first column
       contains the node id and from the second column on are the samples;
       keep in mind that there may be ids that are not in the rows,
       so the sample data matrix needs to be filtered
       to remove rows that do not match any node."
```

## Project Assumptions

- edges between nodes are directional $(source \to target)$

- original numerical values are of type `float16` (extrapolated from `.4f` precision of `test_tumor_samples.tsv`)
  - this typing will be altered inside MPWAE for augmented precision on standardization

- Some nodes and edges are to be ignored as they are irrelevant to the model encoding
  - every node in `tumor_test_samples.tsv` that is not in `metapathway_nodes_2025.tsv` will be ignored and vice versa
  - edges source -> target will be ignored if source is not in input, target is not present in `metapathway_nodes_to_pathways_2025.tsv`

## Architecture

While this is a fairly "simple" autoencoder, some drafts of its architecture have been made. Starting from the Naive approach (where everything is integrated as-is, with little hopes of greatness) up to the current Convolutional MetaPathWay AE with two convolutional layers.

Here is the [Architecture Design folder (Google Drive)](https://drive.google.com/drive/folders/1b_HbaFRjQrZYtTOQplmKU97p9jKFjhyi?usp=drive_link).

> **Personal log.** This design scaling is due to the scarce amout of time I had in my personal day to day life to work on the project, so that I could work a little at a time and improve as I went.
>
>I overcomplicated things at the start so i restarted the project with past knowledge: this is why the model is called MPWAE**2**. The 2 also represents the two added Conv1d layers for compression.
>
> Old MPWAE followed the naive architecture for ease of testing, the scope of this repository doesn't include it.

## What can be improved

> [TL;DR] Everything
>
> At this time $R^{2}$ score signals a positive albeit very small quality of the model in respect to guessing the reconstruction of data based off the average for each node.
>
> The most likely cause is an abuse of (the same) activation function, please look at the following lists to see where improvements are needed exactly.
>
> > Update: ArchTanh being added, the above statements may now be obsolete, some experimentation is required

### Activation Functions

Although hyperbolic tangent is the default choice for this architecture, alternative activation functions may perform better when paired with standardization ($\mu, \sigma$) or normalization $[-1, 1]$ (**standardization is preferred**)

While the number of activation steps follows the standard $HL + 1$, this might be an excessive use of the same two functions (**requires experimental confirmation**)

A custom ArcTanh(x) activation layer may provide optimal results in the decoder, as it effectively reverses prior activations (**implemented, needs practical testing**)

### Generalization

The current architecture can be generalized by introducing a hyperparameter derived from the number of nodes retained post-filtering and the amount of metapathways. This hyperparameter defines the amount of 1D convolutional layers instantiated per model instance for compression purposes.

- Pros
  - adaptable models across various datasets that may add/miss nodes compared to original files.
  - better control over data leakage caused by compression

- Cons: inconsistent encoding representations of metapathways across the different model instances.

An alternative involves increasing complexity to support true graph convolution (see next section).

## Future Architecture Plans

A future model might integrate networkx and torch_geometric libraries to enable full graph convolutional operations.

This architecture could theoretically reconstruct missing node data absent from input, working on the whole knowledge base at once and reconstructing missing data by inference.

This model must be explicitly trained to make this reconstruction reliable. In this case Dropout-based techniques are mandatory to enhance recontruction .
