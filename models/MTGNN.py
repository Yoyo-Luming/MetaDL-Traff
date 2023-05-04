import torch
from torch import nn
import torch.nn.functional as F
import numbers
import numpy as np


class graph_constructor(nn.Module):         # uni-directed: M1M2-M2M1
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
        self.meta_nodevec1 = None
        self.meta_nodevec2 = None
    
    def set_nodevec(self, nodevec1, nodevec2):
        self.meta_nodevec1 = nodevec1
        self.meta_nodevec2 = nodevec2       
        return
        
    def forward(self, idx):
        # print('idx:', idx.shape)
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        if self.meta_nodevec1 is None:
            nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1)) 
            nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
            a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
            adj = F.relu(torch.tanh(self.alpha*a)) # N N 
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1,t1 = adj.topk(self.k,1)          # top-k sparsify
            mask.scatter_(1,t1,s1.fill_(1))     # discretize to {0, 1}?
            adj = adj*mask
        else:
            batch_size = self.meta_nodevec1.shape[0]
            nodevec1 = torch.tanh(self.alpha*self.lin1(self.meta_nodevec1)) 
            nodevec2 = torch.tanh(self.alpha*self.lin1(self.meta_nodevec2)) 
            a = torch.bmm(nodevec1, nodevec2.transpose(1,2))-torch.bmm(nodevec2, nodevec1.transpose(1,2))

            adj = F.relu(torch.tanh(self.alpha*a)) # N N 
            mask = torch.zeros(batch_size, idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1,t1 = adj.topk(self.k,2)          # top-k sparsify
            mask.scatter_(2,t1,s1.fill_(1))     # discretize to {0, 1}?
            adj = adj*mask
        return adj


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        # print('mixprop adj:', adj.shape)
        if len(adj.shape) == 3:
            batch_size = adj.shape[0]
            num_nodes = adj.shape[1]
            adj = adj + torch.eye(num_nodes).expand(batch_size, num_nodes, num_nodes).to(x.device)
            d = adj.sum(2)
            # print('d1:',d.shape)
            d = torch.unsqueeze(d, -1)
            # print('d2:',d.shape)
            # a = adj / d.view(-1, 1)
            a = torch.div(adj, d)
            # print('a:', a.shape)
        else:
            adj = adj + torch.eye(adj.size(0)).to(x.device)
            d = adj.sum(1) # N 
            a = adj / d.view(-1, 1) # N 1
        h = x
        out = [h]
        
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        if len(A.shape)==2:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
        else:
            x = torch.einsum('ncvl,nvw->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)




class MTGNN(nn.Module):
    def __init__(self, 
                 gcn_true, 
                 buildA_true, 
                 gcn_depth, 
                 num_nodes, 
                 device, 
                 predefined_A=None, 
                 static_feat=None, 
                 dropout=0.3, 
                 subgraph_size=20, 
                 node_dim=40, 
                 dilation_exponential=1, 
                 conv_channels=32, 
                 residual_channels=32, 
                 skip_channels=64, 
                 end_channels=128, 
                 seq_length=12, 
                 in_dim=2, 
                 out_dim=12, 
                 layers=3, 
                 propalpha=0.05, 
                 tanhalpha=3, 
                 layer_norm_affline=True,
                 add_meta_adj=False,
                 node_emb_file=None,
                 tod_embedding_dim=24,
                 dow_embedding_dim=7,
                 node_embedding_dim=64,
                 learner_hidden_dim=128,
                 z_dim=32,
                 in_steps=12):
        super(MTGNN, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.add_meta_adj = add_meta_adj
        self.use_meta = self.add_meta_adj
        self.st_embedding_dim = (
            tod_embedding_dim + dow_embedding_dim + node_embedding_dim
        )
        self.learner_hidden_dim = learner_hidden_dim
        self.z_dim = z_dim
        if self.use_meta:
            self.node_embedding = torch.FloatTensor(np.load(node_emb_file)["data"]).to(device)

            self.tod_onehots = torch.eye(24, device=device)
            self.dow_onehots = torch.eye(7, device=device)  
            
            if self.z_dim > 0:
                self.mu = nn.Parameter(torch.randn(num_nodes, z_dim), requires_grad=True)
                self.logvar = nn.Parameter(
                    torch.randn(num_nodes, z_dim), requires_grad=True
                )

                self.mu_estimator = nn.Sequential(
                    nn.Linear(in_steps, 32),
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, z_dim),
                )

                self.logvar_estimator = nn.Sequential(
                    nn.Linear(in_steps, 32),
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, z_dim),
                )    

            if self.add_meta_adj:
                self.adj_learner1 = nn.Sequential(
                    nn.Linear(self.st_embedding_dim + z_dim, learner_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(learner_hidden_dim, node_dim),
                ) 
                self.adj_learner2 = nn.Sequential(
                    nn.Linear(self.st_embedding_dim + z_dim, learner_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(learner_hidden_dim, node_dim),
                )  

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1,1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        batch_size = input.shape[0]

        if self.use_meta:
            tod = input[..., 1]  # (B, T_in, N)
            dow = input[..., 2]  # (B, T_in, N)
            input = input[..., :2]  # (B, T_in, N, 1)
                    # use the last time step to represent the temporal location of the time seires
            x = input[..., :1]
            tod_embedding = self.tod_onehots[(tod[:, -1, :] * 24).long()]  # (B, N, 24)
            dow_embedding = self.dow_onehots[dow[:, -1, :].long()]  # (B, N, 7)
            node_embedding = self.node_embedding.expand(
                batch_size, *self.node_embedding.shape
            )  # (B, N, node_emb_dim)
            
            meta_input = torch.concat(
                [node_embedding, tod_embedding, dow_embedding], dim=-1
            )  # (B, N, st_emb_dim)

            if self.z_dim > 0:
                z_input = x.squeeze(dim=-1).transpose(1, 2)

                mu = self.mu_estimator(z_input)  # (B, N, z_dim)
                logvar = self.logvar_estimator(z_input)  # (B, N, z_dim)

                z_data = self.reparameterize(mu, logvar)  # temporal z (B, N, z_dim)
                z_data = z_data + self.reparameterize(
                    self.mu, self.logvar
                )  # temporal z + spatial z
                
                meta_input = torch.concat(
                    [meta_input, z_data], dim=-1
                )  # (B, N, st_emb_dim+z_dim)

            if self.add_meta_adj:
                meta_nodevec1 = self.adj_learner1(meta_input)
                meta_nodevec2 = self.adj_learner2(meta_input)
                self.gc.set_nodevec(meta_nodevec1, meta_nodevec2)

        input = input.permute(0, 3, 2, 1)
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                if self.add_meta_adj:
                    x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,2))
                else:
                    x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std