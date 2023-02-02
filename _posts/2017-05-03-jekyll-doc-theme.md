---
layout: post
title:  "Jekyll Doc Theme is published!"
author: aksakalli


Introduction to SALIENT
SALIENT is a deep learning framework for graph-based data. It provides efficient data loading, parallel and serial training, and seamless deployment.

Key Features
Efficient graph data loading using FastSampler
Parallel and serial training options
Deployment-friendly
Easy to use with a high-level API
FastSampler
FastSampler is an efficient data loader for graph-based data. It uses pre-computation of neighbor indices to achieve fast indexing and efficient GPU memory usage. To use FastSampler, first create a FastSamplerConfig object and pass the required parameters, including the graph data and batch size. Then, create a FastSampler object using the FastSamplerConfig object and the number of worker threads. Finally, create a DevicePrefetcher object using the FastSampler object and a list of devices (usually GPU devices).

python
Copy code
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
Training
SALIENT provides both serial and parallel training options. The serial training function serial_train takes a model, train core function, device iterator, optimizer, and optional train callback as input. The train core function specifies the forward pass and loss computation for a single batch of data. The device iterator is typically the transferer object created using the FastSampler and DevicePrefetcher objects.

csharp
Copy code
# Example usage

# Train core function
def barebone_train_core(model: torch.nn.Module, batch: PreparedBatch):
    out = model(batch.x, batch.adjs)
    loss = F.nll_loss(out, batch.y)
    loss.backward()
    return loss

# Train model
serial_train(model, barebone_train_core, transferer, optimizer, lr_scheduler)
Conclusion
SALIENT provides a high-level API for efficient graph-based deep learning. With FastSampler and its parallel/serial training options, users can quickly and easily train their graph models on large datasets.


---
A new custom Jekyll theme for documentation and blogging is out. It is ideal for Open Source Software projects to publish under [GitHub Pages](https://pages.github.com).

Your contribution is welcome!
