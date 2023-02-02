---
title: Introduction to SALIENT
permalink: /docs/home/
redirect_from: /docs/index.html
---

SALIENT is a deep learning framework for graph-based data for both single GPU
and multi-machine multi-GPU workloads. SALIENT accelerates existing graph
machine-learning workloads that use PyTorch Geometric by providing efficient
neighborhood sampling and data loading for mini-batches of graph data during
training and inference. 

## Key Features

* Fast dataset representation for graph-learning workloads.
* Efficient neighborhood sampling using SALIENT's FastSampler.
* Pipelining of data movement, neighborhood sampling, and GPU compute during training and inference.
* Easy to use with existing graph-learning codes that employ PyTorch Geometric.

The installation instructions are provided on the [Install](/docs/install)
page.

## FastDataset

The FastDataset provides efficient graph representations for graph-learning
workloads. It supports seamless conversion with existing graph datasets, such
as those provided by the Open Graph Benchmark (OGB).  To load an OGB dataset from a given path,
you can invoke the `FastDataset.from_path(dataset_root, dataset_name)` as shown in the example below.

```python
dataset = FastDataset.from_path('dataset/', 'ogbn-arxiv')
```

In the above example, the ogbn-arxiv dataset will be loaded. If the dataset
does not yet exist, then it will be automatically downloaded from the OGB
repository and converted so that: (1) the graph is undirected; (2) node
features are represented using half-precision floating point values; and, (3)
the graph structure and feature tensors are organized into a compressed-sparse
row format for efficient access. 

For larger datasets, such as the ogbn-papers100M dataset, the conversion
routine can take significant time and require large amounts of memory. To
alleviate this inconvenience, we provide helper scripts for downloaded
pre-converted versions of popular datasets, such as those from OGB.  **PROVIDE
LINK**.

## FastSampler

FastSampler is an efficient data loader for graph-based data. It implements
efficient codes for performing nodewise sampling in message-passing GNN
architectures like GraphSAGE, GAT, and GIN architectures. 

To use FastSampler, first create a FastSamplerConfig object and pass the
required parameters, including the graph data and batch size. Then, create a
FastSampler object using the FastSamplerConfig object and the number of worker
threads. Finally, create a DevicePrefetcher object using the FastSampler object
and a list of devices (usually GPU devices). The DevicePrefetcher performs
prefetching of mini-batch data to overlap data movement with training during
training and inference.


```python
# Example usage

# Create a FastSamplerConfig object
cfg = FastSamplerConfig(
    x=self.dataset.x, y=self.dataset.y.unsqueeze(-1),
    rowptr=self.dataset.rowptr, col=self.dataset.col,
    idx=self.dataset.split_idx['train'],
    batch_size=args.train_batch_size, sizes=args.train_fanouts,
    skip_nonfull_batch=False, pin_memory=True
)

# Create a FastSampler object
sampler = FastSampler(
    args.num_workers, self.train_max_num_batches, cfg
)

# Create a DevicePrefetcher object
devices = [torch.device(type='cuda', index=i)
               for i in range(num_devices_per_node)]
transferer = DevicePrefetcher(devices, iter(sampler))
```

## Training

SALIENT supports both the use of the distributed data parallel module (DDP) and
the data parallel (DP) modules in PyTorch. Generally, we recommend the use of
DDP as it tends to provide better scalability. When using the DDP module, each
training process is allocated one GPU and uses the `serial_train` function ---
i.e., training code that uses only one GPU. 

The serial training function serial_train takes a model, train core function,
device iterator, optimizer, and optional train callback as input. The train
core function specifies the forward pass and loss computation for a single
batch of data. The device iterator is typically the transferer object created
using the FastSampler and DevicePrefetcher objects.

## Example usage

```python
# Train core function
def barebone_train_core(model: torch.nn.Module, batch: PreparedBatch):
    out = model(batch.x, batch.adjs)
    loss = F.nll_loss(out, batch.y)
    loss.backward()
    return loss

# Train model
serial_train(model, barebone_train_core, transferer, optimizer, lr_scheduler)
```

To learn how to use SALIENT's core components to build an end-to-end code that trains a GraphSAGE architecture,
please see the tutorial example for [GraphSAGE tutorial](/docs/GraphSAGE)

## Conclusion

SALIENT provides a high-level API for efficient graph-based deep learning. With
FastSampler and its parallel/serial training options, users can quickly and
easily train their graph models on large datasets.


