---
title: GraphSAGE tutorial using basic components.
permalink: /docs/GraphSAGE/
---

# Training a GraphSAGE model with SALIENT

The following tutorial will walk you through the process of training a
GraphSAGE model on the ogbn-arxiv dataset using SALIENT. For didactic purposes,
the code examples provided will be mostly straight-line code with limited
abstraction so that it is easier to see how to incorporate SALIENT's fast
sampling and prefetching logic into existing codes.



## Step 1: Install SALIENT

Please follow the instructions at INSTALL for installing SALIENT.

## Step 2: Create a workspace.

For this tutorial, we will assume that the code you are writing is in the file
`SALIENT_ROOT/usage_examples/GraphSAGE_example.py` where `SALIENT_ROOT` is the
directory where you cloned the SALIENT repository.

## Step 3: Imports

First, let us import the required python packages.

```python

# PyTorch/PyG Imports
import torch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

# Utilities/Types
from typing import Callable, List, Mapping, Type, Iterable
from tqdm import tqdm

# Import SALIENT's Fast Dataset.
from driver import dataset
from driver.dataset import FastDataset

# Import SALIENT's Fast Training utilities.
from fast_trainer.samplers import *
from fast_trainer.transferers import *
from fast_trainer.concepts import *
from fast_trainer import train, test
from fast_trainer.shufflers import Shuffler
```

## Step 4: Define the GraphSAGE model

The following code implements a basic GraphSAGE model.

```python
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        kwargs = dict(bias=False, normalize=True)
        conv_layer = SAGEConv
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels

        self.convs.append(conv_layer(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(
                hidden_channels, hidden_channels, **kwargs))
        self.convs.append(conv_layer(
            hidden_channels, hidden_channels, **kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            conv.apply(init_weights)

    def forward(self, x, adjs):
        x = x.to(torch.float)
        x = F.normalize(x, dim=-1)
        end_size = adjs[-1][-1][1]
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return torch.log_softmax(x, dim=-1)
```



# Step 5: Generic training and inference functions.

Next we will implement the functions for performing training and batchwise
inference. These functions are generic, in the sense that they are compatible
with both SALIENT's fast sampler and the generic NeighborSampler/NeighborLoader
provided by PyTorch Geometric.


```python
def barebones_train_core(model: torch.nn.Module, batch: PreparedBatch):
    out = model(batch.x, batch.adjs)
    loss = F.nll_loss(out, batch.y)
    loss.backward()
    return loss

def example_train(model: torch.nn.Module,
                 train_core: TrainCore,
                 devit: DeviceIterator,
                 optimizer: torch.optim.Optimizer,
                 cb: Optional[TrainCallback] = None,
                 dataset=None,
                 devices=None) -> None:
    model.train()
    iterator = iter(devit)
    while True:
        try:
            inp, = next(iterator)
        except StopIteration:
            break
        optimizer.zero_grad()
        result = train_core(model, inp)
        optimizer.step()
        if cb is not None:
            cb([inp], [result])
```

The `example_train` function takes a model, a function `train_core` to compute
the loss function, an iterator `devit` which will iterate over sampled
mini-batches, an optimizer `optimizer`, and a callback function `cb` that will
track progress during an epoch.


For batchwise inference, we have a separate function `batchwise_test` which we provide below:

```python

@torch.no_grad()
def batchwise_test(model, dataset, test_fanouts, test_batch_size, sets=None) -> Mapping[str, float]:
    model.eval()

    if sets is None:
        sets = dataset.split_idx

    results = {}

    for name in sets:
        local_fanouts = test_fanouts
        local_batchsize = test_batch_size

        cfg = FastSamplerConfig(
            x=dataset.x, y=dataset.y.unsqueeze(-1),
            rowptr=dataset.rowptr, col=dataset.col,
            idx=dataset.split_idx[name],
            batch_size=local_batchsize,
            sizes=local_fanouts,
            skip_nonfull_batch=False, pin_memory=True
        )
        loader = FastSampler(20, 40, cfg)
        devit = DevicePrefetcher(devices, iter(loader))

        pbar = tqdm(total=cfg.idx.numel())
        pbar.set_description(f'Batchwise eval')

        def cb(batch):
            pbar.update(batch.batch_size)

        with Timer((name, 'Compute')) as timer:
            result = test.batchwise_test(
                model, len(loader), devit, cb)
            timer.stop()
            pbar.close()
            del pbar
        results[name] = result[0] / result[1]

    return results
```

The `batchwise_test` function takes in a model, dataset, the fanouts used
during inference, the batch size used during inference, and a list of sets
(e.g., a subset of ['train', 'valid', 'test']) of vertices on which to perform
inference. 

### Extra information: Implementation of `test.batchwise_test`

The `test.batchwise_test` function implementation is located in
`fast_sampler/test.py`. The source code is provided for your reference below.
The batchwise test code provided in `fast_sampler/test.py` is generally faster
than the logic provided within the OGB library.

```python
@torch.no_grad()
def batchwise_test(model: torch.nn.Module,
                   num_batches: int,
                   devit: DeviceIterator,
                   cb: TestCallback = None):
    model.eval()

    device, = devit.devices

    results = torch.empty(num_batches, dtype=torch.long, pin_memory=True)
    total = 0

    for i, inputs in enumerate(devit):
        inp, = inputs

        out = model(inp.x, inp.adjs)
        out = out.argmax(dim=-1, keepdim=True).reshape(-1)
        correct = (out == inp.y).sum()
        results[i].copy_(correct, non_blocking=True)
        total += inp.batch_size

        if cb is not None:
            cb(inp)

    torch.cuda.current_stream(device).synchronize()

    return results.sum().item(), total
```



## Step 6: Load the dataset

You can load the dataset using the FastDataset method `FastDataset.from_path` which will take in a `dataset_root` path
to a directory on your filesystem where you store your OGB datasets. If the dataset does not already exist within the `dataset_root`
directory, the `FastDataset.from_path` function will automatically download it from OGB and perform the following conversions: (1) make the graph undirected; (2) reduce the precision of the node features to half-precision floating point; and, (3) convert the graph into CSR (compressed sparse rows) format. For larger datasets, like ogbn-papers100M, the process of performing the dataset conversion can be time consuming and require the use of a machine with a large amount of memory. For larger datasets, we recommend you predownload the converted datasets by following the instructions in INSERTLINK.

The code for loading the ogbn-arxiv dataset is provided below:

```python 
dataset = FastDataset.from_path("dataset/", 'ogbn-arxiv')
```



## Step 7: Create FastSamplerConfig, FastSampler, and Shuffler

The next step is to create a FastSampler object for the dataset. The
FastSampler class accepts as an argument a FastSamplerConfig object that
provides information about the dataset, and parameters for the sampling algorithm.

```python
cfg = FastSamplerConfig(
    x=dataset.x, y=dataset.y.unsqueeze(-1),
    rowptr=dataset.rowptr, col=dataset.col,
    idx=dataset.split_idx['train'],
    batch_size=1024, sizes=[15,10,5],
    skip_nonfull_batch=False, pin_memory=True
)
devices = ['cuda:0']

train_loader = FastSampler(24, 48, cfg)

train_shuffler = Shuffler(dataset.split_idx['train'])
```

In this example, the FastSampler is configured to operate on the 'train' subset
of the ogbn-arxiv nodes with a batchsize of 1024 and sampling fanouts
[15,10,5]. 

## Step 8: Training loop.

Putting all of these pieces together, we can implement a basic loop that trains the GraphSAGE model using SALIENT over multiple epochs.

```python

def cb(inputs, results):
    pbar.update(sum(batch.batch_size for batch in inputs))

for epoch in range(0,10):
    pbar = tqdm(total=train_loader.idx.numel())
    pbar.set_description(f'Epoch {epoch}')
    train_shuffler.set_epoch(epoch)
    train_loader.idx = train_shuffler.get_idx()
    train_transferer = DevicePrefetcher(devices, iter(train_loader))
    example_train(model, barebones_train_core, train_transferer, optimizer, cb)
    pbar.close()
    batchwise_test(model, dataset, [20,20,20], 1024, sets=['valid'])

```

Note that the iterator provided to the `example_train` function is a
DevicePrefetcher which implements the prefetching logic in SALIENT to overlap
data transfers with computation.

The output of this program on am AWS g5.8xlarge instance with 1 GPU is the following:

```bash
(salient) ubuntu@queue1-dy-g58xlarge-1:~/SALIENT_artifact$ python -m usage_examples.GraphSAGE_example                                                                                                                                                                                                            WARNING:root:The OGB package is out of date. Your version is 1.3.2, while the latest version is 1.3.5.                                                                                                                                                                                                           Epoch 0: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 90941/90941 [00:01<00:00, 52430.41it/s]                                                                                                                                                        ('train', 0) took 1.336423033 sec                                                                                                                                                                                                                                                                                Batchwise eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29799/29799 [00:00<00:00, 32179.84it/s]                                                                                                                                                        ('valid', 'Compute') took 0.925837187 sec
Epoch 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 90941/90941 [00:00<00:00, 128704.98it/s]                                                                                                                                                        ('train', 1) took 0.57699822 sec
Batchwise eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29799/29799 [00:00<00:00, 39879.65it/s]                                                                                                                                                        ('valid', 'Compute') took 0.747067687 sec
Epoch 2: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 90941/90941 [00:00<00:00, 135050.50it/s]                                                                                                                                                        ('train', 2) took 0.576809447 sec
Batchwise eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29799/29799 [00:00<00:00, 97823.13it/s]                                                                                                                                                        ('valid', 'Compute') took 0.304427883 sec
Epoch 3: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 90941/90941 [00:00<00:00, 131926.86it/s]                                                                                                                                                        ('train', 3) took 0.577493247 sec
Batchwise eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29799/29799 [00:00<00:00, 95702.01it/s]                                                                                                                                                        ('valid', 'Compute') took 0.311192139 sec
Epoch 4: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 90941/90941 [00:00<00:00, 130905.61it/s]                                                                                                                                                        ('train', 4) took 0.573180366 sec
Batchwise eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29799/29799 [00:00<00:00, 99881.94it/s]                                                                                                                                                        ('valid', 'Compute') took 0.298063962 sec
Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 90941/90941 [00:00<00:00, 139676.20it/s]                                                                                                                                                        ('train', 5) took 0.574929961 sec
Batchwise eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 29799/29799 [00:00<00:00, 103520.99it/s]                                                                                                                                                        ('valid', 'Compute') took 0.287698304 sec
Epoch 6: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 90941/90941 [00:00<00:00, 140022.51it/s]                                                                                                                                                        ('train', 6) took 0.572745331 sec
Batchwise eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29799/29799 [00:00<00:00, 95111.46it/s]                                                                                                                                                        ('valid', 'Compute') took 0.313122667 sec
Epoch 7: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 90941/90941 [00:00<00:00, 140433.33it/s]                                                                                                                                                        ('train', 7) took 0.575713323 sec
Batchwise eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29799/29799 [00:00<00:00, 82880.03it/s]                                                                                                                                                        ('valid', 'Compute') took 0.359362366 sec
Epoch 8: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 90941/90941 [00:00<00:00, 132821.02it/s]                                                                                                                                                        ('train', 8) took 0.572992694 sec
Batchwise eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29799/29799 [00:00<00:00, 89119.67it/s]                                                                                                                                                        ('valid', 'Compute') took 0.334195696 sec
Epoch 9: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 90941/90941 [00:00<00:00, 139386.85it/s]                                                                                                                                                        ('train', 9) took 0.568741793 sec
Batchwise eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 29799/29799 [00:00<00:00, 91845.87it/s]                                                                                                                                                        ('valid', 'Compute') took 0.324283246 sec
```

## Complete code example

The complete code example is located in the SALIENT repository at `usage_examples/GraphSAGE_example.py`. For reference, the full code example is also provided below.

```python
import torch
from tqdm import tqdm

from typing import Callable, List, Mapping, Type, Iterable
from driver import dataset
from driver.dataset import FastDataset
#from fast_trainer.utils import Timer, CUDAAggregateTimer, append_runtime_stats, start_runtime_stats_epoch
from fast_trainer.samplers import *
from fast_trainer.transferers import *
from fast_trainer.concepts import *
from fast_trainer import train, test
from fast_trainer.shufflers import Shuffler

from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        kwargs = dict(bias=False, normalize=True)
        conv_layer = SAGEConv
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels

        self.convs.append(conv_layer(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(
                hidden_channels, hidden_channels, **kwargs))
        self.convs.append(conv_layer(
            hidden_channels, hidden_channels, **kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            conv.apply(init_weights)

    def forward(self, x, adjs):
        x = x.to(torch.float)
        x = F.normalize(x, dim=-1)
        end_size = adjs[-1][-1][1]
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return torch.log_softmax(x, dim=-1)

@torch.no_grad()
def batchwise_test(model, dataset, test_fanouts, test_batch_size, sets=None) -> Mapping[str, float]:
    model.eval()

    if sets is None:
        sets = dataset.split_idx

    results = {}

    for name in sets:
        local_fanouts = test_fanouts
        local_batchsize = test_batch_size

        cfg = FastSamplerConfig(
            x=dataset.x, y=dataset.y.unsqueeze(-1),
            rowptr=dataset.rowptr, col=dataset.col,
            idx=dataset.split_idx[name],
            batch_size=local_batchsize,
            sizes=local_fanouts,
            skip_nonfull_batch=False, pin_memory=True
        )
        loader = FastSampler(20, 40, cfg)
        devit = DevicePrefetcher(devices, iter(loader))

        pbar = tqdm(total=cfg.idx.numel())
        pbar.set_description(f'Batchwise eval')

        def cb(batch):
            pbar.update(batch.batch_size)

        with Timer((name, 'Compute')) as timer:
            result = test.batchwise_test(
                model, len(loader), devit, cb)
            timer.stop()
            pbar.close()
            del pbar
        results[name] = result[0] / result[1]

    return results
def barebones_train_core(model: torch.nn.Module, batch: PreparedBatch):
    out = model(batch.x, batch.adjs)
    loss = F.nll_loss(out, batch.y)
    loss.backward()
    return loss

def example_train(model: torch.nn.Module,
                 train_core: TrainCore,
                 devit: DeviceIterator,
                 optimizer: torch.optim.Optimizer,
                 cb: Optional[TrainCallback] = None,
                 dataset=None,
                 devices=None) -> None:
    model.train()
    iterator = iter(devit)
    while True:
        try:
            inp, = next(iterator)
        except StopIteration:
            break
        optimizer.zero_grad()
        result = train_core(model, inp)
        optimizer.step()
        if cb is not None:
            cb([inp], [result])


def get_dataset(dataset_name, root):
    assert dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']
    return FastDataset.from_path(root, dataset_name)



dataset = FastDataset.from_path("dataset/", 'ogbn-arxiv')

# TODO: Add 1D version of serial_idx kernel
cfg = FastSamplerConfig(
    x=dataset.x, y=dataset.y.unsqueeze(-1),
    rowptr=dataset.rowptr, col=dataset.col,
    idx=dataset.split_idx['train'],
    batch_size=1024, sizes=[15,10,5],
    skip_nonfull_batch=False, pin_memory=True
)
devices = ['cuda:0']

train_loader = FastSampler(24, 48, cfg)

train_shuffler = Shuffler(dataset.split_idx['train'])
model = SAGE(dataset.num_features, 256, dataset.num_classes, 3).to(devices[0])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def cb(inputs, results):
    pbar.update(sum(batch.batch_size for batch in inputs))

for epoch in range(0,10):
    pbar = tqdm(total=train_loader.idx.numel())
    pbar.set_description(f'Epoch {epoch}')
    train_shuffler.set_epoch(epoch)
    train_loader.idx = train_shuffler.get_idx()
    train_transferer = DevicePrefetcher(devices, iter(train_loader))
    example_train(model, barebones_train_core, train_transferer, optimizer, cb)
    pbar.close()
    batchwise_test(model, dataset, [20,20,20], 1024, sets=['valid'])

```

 



