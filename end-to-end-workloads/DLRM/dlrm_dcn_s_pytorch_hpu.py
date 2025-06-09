import argparse
import sys
import time
from itertools import accumulate
import math
from math import log2
import subprocess
import multiprocessing

# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import habana_frameworks.torch.hpu as hthpu
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.kernels import EmbeddingBag
import habana_frameworks.torch.hpex.kernels.fbgemm as fbgemm

class DLRM_DCN_Net(nn.Module):
    def create_mlp(self, ln, activation_skip_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True).to(self.device)

            # initialize the weights
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=False).to(self.device)
            LL.bias.data = torch.tensor(bt, requires_grad=False).to(self.device)

            layers.append(LL)

            # construct sigmoid or relu operator
            if i == activation_skip_layer:
                pass
            else:
                layers.append(nn.ReLU().to(self.device))
        
        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers).to(self.device)
    
    def create_crossnet(self, in_features: int, num_layers: int, low_rank: int = 1):
        assert low_rank >= 1, "Low rank must be larger or equal to 1"

        W_kernels = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(in_features, low_rank, device=self.device)
                    )
                )
                for i in range(num_layers)
            ]
        )
        V_kernels = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(low_rank, in_features, device=self.device)
                    )
                )
                for i in range(num_layers)
            ]
        )
        bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features, device=self.device)))
                for i in range(num_layers)
            ]
        )

        return W_kernels, V_kernels, bias
        
    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def create_emb(self, m, ln):
        # emb_l = nn.ModuleList()
        emb_l = []
        for i in range(0, ln.size):
            n = ln[i]
            # construct embedding operator
            # EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False)
            # initialize embeddings
            if self.validation:
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
            else:
                W = np.arange(n * m, dtype=np.float32).reshape(n, m)
            # approach 1
            # EE.weight.data = torch.tensor(W, requires_grad=False)
            # approach 2
            # EE.weight.data.copy_(torch.tensor(W))
            # approach 3
            # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            emb_l.append(torch.tensor(W, requires_grad=False).to(self.device))
        return emb_l

    def apply_emb(self, lS_o, lS_i, emb_l, num_tpc):
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            E = emb_l[k]
            V = self.embedding_bag(E, sparse_index_group_batch, 
                                    sparse_offset_group_batch, num_tpc)
            ly.append(V)
        return ly

    def create_emb_fbgemm(self, m, ln):
        embedding_specs=[(ln[i], m) for i in range(ln.size)]
        (rows, dims) = zip(*embedding_specs)
        T_ = len(embedding_specs)
        feature_table_map = list(range(T_))
        weights_offsets = [0] + list(
            accumulate([row * dim for (row, dim) in embedding_specs])
        )    
        weights_offsets = [weights_offsets[t] for t in feature_table_map]
        weights_offsets = torch.tensor(weights_offsets, dtype=torch.int64)
        
        feature_dims = [dims[t] for t in feature_table_map]
        D_offsets = [0] + list(accumulate(feature_dims))
        D_offsets = torch.tensor(D_offsets, dtype=torch.int32)
        total_D = D_offsets[-1]

        emb_l = []
        for i in range(0, ln.size):
            n = ln[i]
            if self.validation:
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n*m,)
                ).astype(np.float32)
            else:
                W = np.arange(n * m, dtype=np.float32)
            emb_l.append(torch.tensor(W, requires_grad=False))
        
        weights = torch.stack(emb_l, dim=0).flatten().reshape(ln.size * n * m).contiguous().to(self.device)

        return weights, weights_offsets, D_offsets, total_D

    def create_emb_custom_fbgemm(self, m, ln):
        embedding_specs=[(ln[i], m) for i in range(ln.size)]
        (rows, dims) = zip(*embedding_specs)
        weight_width_offset = [0] + list(accumulate([row for row in rows]))
        weight_width_offset = weight_width_offset[:-1]
        weight_width_offset = torch.tensor(weight_width_offset, dtype=torch.int32).to(self.device)

        emb_l = []
        for i in range(0, ln.size):
            n = ln[i]
            if self.validation:
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n*m,)
                ).astype(np.float32)
            else:
                W = np.arange(n * m, dtype=np.float32)
            emb_l.append(torch.tensor(W, requires_grad=False))
        
        # Here, we assume that all embedding tables have the same number of rows
        weights = torch.stack(emb_l, dim=0).flatten().reshape(self.num_tables * rows[0], self.emb_dim).to(self.device)

        return weights, weight_width_offset

    def apply_emb_fbgemm(self, offsets, indices):
        output = fbgemm.split_embedding_codegen_lookup_function(self.weights, self.weights_offsets, self.D_offsets, self.total_D, indices, offsets, 0)
        return output

    def apply_emb_custom_fbgemm(self, offsets, indices, num_tpc, num_tables):
        output = self.embedding_bag(self.weights, self.weight_width_offset, indices, offsets, num_tpc, num_tables)
        return output

    def interact_features(self, x, ly):
        # concatenate dense and sparse features
        (batch_size, d) = x.shape
        if self.use_fbgemm or self.use_custom_fbgemm:
            combined_values = torch.cat((x, ly), dim=1)
        else:
            sparse_features = torch.stack(ly, dim=1)  # Shape: (batch_size, num_features, d)
            combined_values = torch.cat((x.unsqueeze(1), sparse_features), dim=1).reshape([batch_size, -1]) 

        x_l = combined_values

        for layer in range(self.dcn_num_layers):
            x_l_v = torch.nn.functional.linear(x_l, self.dcn_V_kernels[layer])
            x_l_w = torch.nn.functional.linear(x_l_v, self.dcn_W_kernels[layer])
            x_l = combined_values * (x_l_w + self.dcn_bias[layer]) + x_l  # (B, N)

        return x_l

    def __init__(
        self, 
        emb_dim, 
        ln_bot,
        ln_top,
        ln_emb,
        dcn_num_layers,
        dcn_low_rank_dim,
        activation_skip_layer_bot=-1,
        activation_skip_layer_top=-1,
        device='cpu',
        validation=False,
        use_fbgemm=False,
        use_custom_fbgemm=False,
        latency_breakdown=False):
        super(DLRM_DCN_Net, self).__init__()
        

        # Variables
        self.device = device
        self.emb_dim = emb_dim
        self.num_tables = len(ln_emb)
        self.dcn_num_layers = dcn_num_layers
        self.dcn_low_rank_dim = dcn_low_rank_dim
        self.latency_breakdown = latency_breakdown
        self.validation = validation
        self.use_fbgemm = use_fbgemm
        self.use_custom_fbgemm = use_custom_fbgemm

        if self.latency_breakdown:
            self.bot_mlp_tic = ht.hpu.Event(enable_timing=True)
            self.bot_mlp_toc = ht.hpu.Event(enable_timing=True)
            self.emb_tic = ht.hpu.Event(enable_timing=True)
            self.emb_toc = ht.hpu.Event(enable_timing=True)
            self.interaction_tic = ht.hpu.Event(enable_timing=True)
            self.interaction_toc = ht.hpu.Event(enable_timing=True)
            self.top_mlp_tic = ht.hpu.Event(enable_timing=True)
            self.top_mlp_toc = ht.hpu.Event(enable_timing=True)

        # Bottom MLP
        self.bot_mlp = self.create_mlp(ln_bot, -1)

        # Top MLP
        self.top_mlp = self.create_mlp(ln_top, ln_top.size - 2)

        # EmbeddingBag layer
        if self.use_fbgemm:
            self.weights, self.weights_offsets, self.D_offsets, self.total_D = self.create_emb_fbgemm(emb_dim, ln_emb)
        elif self.use_custom_fbgemm:
            from custom_table_batched_embedding_bag_sum import CustomTableBatchedEmbeddingBagSum
            self.embedding_bag = CustomTableBatchedEmbeddingBagSum()
            self.weights, self.weight_width_offset = self.create_emb_custom_fbgemm(emb_dim, ln_emb)
        else:
            from custom_embedding_bag_sum import CustomEmbeddingBagSum
            self.embedding_bag = CustomEmbeddingBagSum()
            self.emb_l = self.create_emb(emb_dim, ln_emb)

        # CrossNet layer
        self.in_features=(self.num_tables + 1) * self.emb_dim
        self.dcn_W_kernels, self.dcn_V_kernels, self.dcn_bias = self.create_crossnet(self.in_features, dcn_num_layers, dcn_low_rank_dim)

    def forward(self, dense_input, indices, offsets, num_tpc=24):
        # Pass through linear layers with ReLU activation
        if self.latency_breakdown:
            self.bot_mlp_tic.record()
        x = self.apply_mlp(dense_input, self.bot_mlp)
        if self.latency_breakdown:
            self.bot_mlp_toc.record()
            self.bot_mlp_toc.wait()
            ht.hpu.synchronize()

        # Embedding bag layer output
        # emb_result = []
        if self.latency_breakdown:
            self.emb_tic.record()
        # Approach 1: custom embeddingbag
        if self.use_fbgemm:
            emb_result = self.apply_emb_fbgemm(offsets, indices)
        elif self.use_custom_fbgemm:
            emb_result = self.apply_emb_custom_fbgemm(offsets, indices, num_tpc, self.num_tables)
        else:
            emb_result = self.apply_emb(offsets, indices, self.emb_l, num_tpc)
        # Approach 2: torch embeddingbag
        # emb_result = self.apply_emb_torch_bag(offsets, indices, self.emb_l)
        if self.latency_breakdown:
            self.emb_toc.record()
            self.emb_toc.wait()
            ht.hpu.synchronize()

        # Interaction
        if self.latency_breakdown:
            self.interaction_tic.record()
        # R = torch.cat([x] + emb_result, dim=1)
        R = self.interact_features(x, emb_result)
        if self.latency_breakdown:
            self.interaction_toc.record()
            self.interaction_toc.wait()
            ht.hpu.synchronize()


        # Top MLP
        if self.latency_breakdown:
            self.top_mlp_tic.record()
        z = self.apply_mlp(R, self.top_mlp)
        if self.latency_breakdown:
            self.top_mlp_toc.record()
            self.top_mlp_toc.wait()
            ht.hpu.synchronize()

        if self.latency_breakdown:
            print('bot_mlp:', self.bot_mlp_tic.elapsed_time(self.bot_mlp_toc) /1000)
            print('emb:', self.emb_tic.elapsed_time(self.emb_toc) /1000)
            print('interaction:', self.interaction_tic.elapsed_time(self.interaction_toc) /1000)
            print('top_mlp:', self.top_mlp_tic.elapsed_time(self.top_mlp_toc) /1000)

        return z

def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value

def inference(
    args,
    dlrm,
    test_ld,
    device,
    use_hpu,
):
    # memcpy_tics = [hthpu.Event(enable_timing=True) for _ in range(len(test_ld))]
    # memcpy_tocs = [hthpu.Event(enable_timing=True) for _ in range(len(test_ld))]

    # tics = [hthpu.Event(enable_timing=True) for _ in range(len(test_ld))]
    # tocs = [hthpu.Event(enable_timing=True) for _ in range(len(test_ld))]

    memcpy_tic = hthpu.Event(enable_timing=True)
    memcpy_toc = hthpu.Event(enable_timing=True)

    tic = hthpu.Event(enable_timing=True)
    toc = hthpu.Event(enable_timing=True)

    memcpy_mss = []
    mss = []
    Z_test_list = []

    X_test_static = test_ld[0][0].to(device)
    if args.use_fbgemm or args.use_custom_fbgemm:
        lS_o_test_static = test_ld[0][1].to(device)
        lS_i_test_static = test_ld[0][2].to(device)
    else:
        lS_o_test_static = [S_o.to(device) for S_o in test_ld[0][1]]
        lS_i_test_static = [S_i.to(device) for S_i in test_ld[0][2]]
    hthpu.synchronize()

    # warm up for graph
    Z_test_static = dlrm(X_test_static, lS_i_test_static, lS_o_test_static)
    graph = htcore.hpu.HPUGraph()
    with htcore.hpu.graph(graph):
        Z_test_static = dlrm(X_test_static, lS_i_test_static, lS_o_test_static)

    start_time = time.time()
    while time.time() - start_time < args.power_batch_duration:
        for i, testBatch in enumerate(test_ld):
            # early exit if nbatches was set by the user and was exceeded
            if nbatches > 0 and i >= nbatches:
                break
                
            X_test, lS_o_test, lS_i_test = testBatch[0], testBatch[1], testBatch[2]

            # forward pass
            if args.latency_breakdown:
                memcpy_tic.record()
                
            X_test_static.copy_(X_test.to(device))

            if args.use_fbgemm or args.use_custom_fbgemm:
                lS_o_test_static.copy_(lS_o_test.to(device))
                lS_i_test_static.copy_(lS_i_test.to(device))
            else:
                for j in range(len(lS_o_test_static)):
                    lS_o_test_static[j].copy_(lS_o_test[j].to(device))

                for j in range(len(lS_i_test_static)):
                    lS_i_test_static[j].copy_(lS_i_test[j].to(device))

            if args.latency_breakdown:
                memcpy_toc.record()
                # memcpy_toc.wait()
            hthpu.synchronize()

            if args.latency_breakdown:
                memcpy_mss.append(memcpy_tic.elapsed_time(memcpy_toc))

            with torch.no_grad():
                tic.record()
                graph.replay()
                toc.record()
            htcore.mark_step()
            hthpu.synchronize()

            mss.append(tic.elapsed_time(toc))

            if args.validation:
                Z_test_cpu = Z_test_static.to('cpu')
                Z_test_list.append(Z_test_cpu)

    hthpu.synchronize() # Synchronize after finishing
    
    for i in range(len(mss)):
        print(i, 'th pass:', mss[i] / 1000)
        if args.latency_breakdown:
            print('memcpy:', memcpy_mss[i]/1000)
    if args.validation:
        torch.save(Z_test_list, 'DLRM_DCN_GRAPH_batchsize'+str(args.test_mini_batch_size)+'_result_hpu.pt')


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--dcn-num-layers", type=int, default=-1)
    parser.add_argument("--dcn-low-rank-dim", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation",
        type=str,
        choices=["random", "dataset", "random_gaudi"],
        default="random",
    )  # synthetic, dataset
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--round-targets", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # hpu
    parser.add_argument("--use-hpu", action="store_true", default=False)
    # yunjae
    parser.add_argument("--test-load", action="store_true", default=False)
    parser.add_argument("--test-file-name", type=str, default="")
    parser.add_argument("--latency-breakdown", action="store_true", default=False)
    parser.add_argument("--validation", action="store_true", default=False)
    parser.add_argument("--use-fbgemm", action="store_true", default=False)
    parser.add_argument("--use-custom-fbgemm", action="store_true", default=False)

    global args
    global nbatches
    global nbatches_test
    global writer
    args = parser.parse_args()

    if args.dataset_multiprocessing:
        assert sys.version_info[0] >= 3 and sys.version_info[1] > 7, (
            "The dataset_multiprocessing "
            + "flag is susceptible to a bug in Python 3.7 and under. "
            + "https://github.com/facebookresearch/dlrm/issues/172"
        )

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size

    use_hpu = args.use_hpu and hthpu.is_available()

    if use_hpu:
        nhpus = 1
        device = torch.device("hpu")
        print("Using {} HPU(s)...".format(nhpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data
    if args.data_generation == "dataset":
        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        table_feature_map = {idx: idx for idx in range(len(train_data.counts))}
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(
                list(
                    map(
                        lambda x: x if x < args.max_ind_range else args.max_ind_range,
                        ln_emb,
                    )
                )
            )
        else:
            ln_emb = np.array(ln_emb)
        m_den = train_data.m_den
        ln_bot[0] = m_den
    elif args.data_generation == "random_gaudi":
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]

        if args.test_load:
            test_ld = torch.load(args.test_file_name)
            nbatches = args.num_batches
            nbatches_test = len(test_ld)
        else:
            train_data, train_ld, test_data, test_ld = dp.make_random_data_and_loader(
                args, ln_emb, m_den
            )
            nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
            nbatches_test = len(test_ld)
            torch.save(test_ld, args.test_file_name)

    args.ln_emb = ln_emb.tolist()

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    num_int = num_fea * m_den_out
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if m_spa != m_den_out:
        sys.exit(
            "ERROR: arch-sparse-feature-size "
            + str(m_spa)
            + " does not match last dim of bottom mlp "
            + str(m_den_out)
        )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    global dlrm
    dlrm = DLRM_DCN_Net(
        m_spa,
        ln_bot,
        ln_top,
        ln_emb,
        dcn_num_layers = args.dcn_num_layers,
        dcn_low_rank_dim = args.dcn_low_rank_dim,
        activation_skip_layer_bot=-1,
        activation_skip_layer_top=ln_top.size - 2,
        use_fbgemm=args.use_fbgemm,
        use_custom_fbgemm=args.use_custom_fbgemm,
        validation=args.validation,
        device='hpu',
        latency_breakdown=args.latency_breakdown
    )
    dlrm = dlrm.eval()
    if use_hpu:
        dlrm = dlrm.to(device)
        ht.torch.backends.cuda.matmul.allow_tf32 = True
        ht.torch.backends.cudnn.allow_tf32 = True

    # ext_dist.barrier()
    if args.inference_only:
        print("Testing for inference only")
        inference(
            args,
            dlrm,
            test_ld,
            device,
            use_hpu,
        )

if __name__ == "__main__":
    run()
