import torch
import numpy as np

np.random.seed(1993)
torch.manual_seed(1993)

## DLRM DCN RM1 for FBGEMM ##
# Define constants
num_emb_table = 10
num_emb = 1000000
batchsizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
num_lookups = 10
iterations = 101
bot_mlp = 512

for batchsize in batchsizes:
    # Initialize list to store all batches
    all_batches = []
    # Generate and store data for 101 iterations
    print(batchsize, 'starts')
    for i in range(iterations):
        dense_input = torch.rand((batchsize, bot_mlp), dtype=torch.float32)
        indice_hpu = []
        for t in range(num_emb_table):
            indice = torch.randint(0, num_emb, (batchsize * num_lookups,), dtype=torch.int)
            indice_hpu.append(indice)
            # indice_hpu.append(torch.tensor(indice, dtype=torch.int))

        offset_hpu = torch.tensor([num_lookups * x for x in range(num_emb_table * batchsize + 1)], dtype=torch.int)
        indice_hpu = torch.stack(indice_hpu).flatten().to(torch.int)
        # Append each iteration's batch to the list
        all_batches.append((dense_input, offset_hpu, indice_hpu))

    # Save the entire dataset (all 101 batches) into a single file
    torch.save(all_batches, "dataset/DLRM_DCN_RM1_FBGEMM_random_batchsize"+str(batchsize)+".pt")

## DLRM DCN RM2 ##
# Define constants
num_emb_table = 20
num_emb = 1000000
batchsizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
num_lookups = 100
iterations = 101
bot_mlp = 256

for batchsize in batchsizes:
    # Initialize list to store all batches
    all_batches = []
    # Generate and store data for 101 iterations
    print(batchsize, 'starts')
    for i in range(iterations):
        dense_input = torch.rand((batchsize, bot_mlp), dtype=torch.float32)
        indice_hpu = []
        for t in range(num_emb_table):
            indice = torch.randint(0, num_emb, (batchsize * num_lookups,), dtype=torch.int)
            indice_hpu.append(indice)
            # indice_hpu.append(torch.tensor(indice, dtype=torch.int))

        offset_hpu = torch.tensor([num_lookups * x for x in range(num_emb_table * batchsize + 1)], dtype=torch.int)
        indice_hpu = torch.stack(indice_hpu).flatten().to(torch.int)
        # Append each iteration's batch to the list
        all_batches.append((dense_input, offset_hpu, indice_hpu))

    # Save the entire dataset (all 101 batches) into a single file
    torch.save(all_batches, "dataset/DLRM_DCN_RM2_FBGEMM_random_batchsize"+str(batchsize)+".pt")


