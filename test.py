import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import math, random, sys
from collections import deque
from jtnn import *
import rdkit
import rdkit.Chem as Chem

train_path='data/train.txt'
test_path='data/test.txt'
vocab_path='data/vocab.txt'
save_path='molvae/pre_model/'
model_path=None

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

def pretrain_vae(hidden_size, latent_size, depth, batch_size, device):
    vocab = [x.strip("\r\n ") for x in open(vocab_path)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depth, device)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    scheduler.step()

    dataset = MoleculeDataset(train_path)

    MAX_EPOCH = 3
    PRINT_ITER = 1

    for epoch in xrange(MAX_EPOCH):
        print "new epoch:", epoch
        sys.stdout.flush()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: x,
                                drop_last=True)

        word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0

        for it, batch in enumerate(dataloader):
            for mol_tree in batch:
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
                        node.cand_mols.append(node.label_mol)

            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta=0)
            loss.backward()
            optimizer.step()

            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc

            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                steo_acc = steo_acc / PRINT_ITER * 100
                print "batchid:",str(it)
                print "KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f" % (
                kl_div, word_acc, topo_acc, assm_acc, steo_acc)
                word_acc, topo_acc, assm_acc, steo_acc = 0, 0, 0, 0
                sys.stdout.flush()
                scheduler.step()
        scheduler.step()
        print "learning rate: %.6f" % scheduler.get_lr()[0]
        torch.save(model.state_dict(), save_path + "/model.iter-" + str(epoch))


def sample(nsample, hidden_size, latent_size, depth, device, model_path):    
    vocab = [x.strip("\r\n ") for x in open(vocab_path)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depth, device)
    load_dict = torch.load(model_path,map_location=device)
    missing = {k: v for k, v in model.state_dict().items() if k not in load_dict}
    load_dict.update(missing) 
    model.load_state_dict(load_dict)

    torch.manual_seed(0)
    for i in xrange(nsample):
        print model.sample_prior()

def reconstruct(hidden_size,latent_size,depth,device,model_path):
    vocab = [x.strip("\r\n ") for x in open(vocab_path)] 
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, hidden_size, latent_size, depth, device)
    model.load_state_dict(torch.load(model_path,map_location=device))

    data = []
    with open(test_path) as f:
        for line in f:
            s = line.strip("\r\n ").split()[0]
            data.append(s)

    acc = 0.0
    tot = 0
    for smiles in data:
        mol = Chem.MolFromSmiles(smiles)
        smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        
        dec_smiles = model.reconstruct(smiles3D)
        if dec_smiles == smiles3D:
            acc += 1
        tot += 1
        print acc / tot
        """
        dec_smiles = model.recon_eval(smiles3D)
        tot += len(dec_smiles)
        for s in dec_smiles:
            if s == smiles3D:
                acc += 1
        print acc / tot
        """

def train_vae(hidden_size,latent_size,depth,beta, anneal, lr,batch_size, device):    
    vocab = [x.strip("\r\n ") for x in open(vocab_path)] 
    vocab = Vocab(vocab)
    model = JTNNVAE(vocab, hidden_size, latent_size, depth, device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path,map_location=device))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
    scheduler.step()

    dataset = MoleculeDataset(train_path)

    MAX_EPOCH = 7
    PRINT_ITER = 20

    for epoch in xrange(MAX_EPOCH):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x:x, drop_last=True)

        word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0

        for it, batch in enumerate(dataloader):
            for mol_tree in batch:
                for node in mol_tree.nodes:
                    if node.label not in node.cands:
                        node.cands.append(node.label)
                        node.cand_mols.append(node.label_mol)

            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta)
            loss.backward()
            optimizer.step()

            word_acc += wacc
            topo_acc += tacc
            assm_acc += sacc
            steo_acc += dacc

            if (it + 1) % PRINT_ITER == 0:
                word_acc = word_acc / PRINT_ITER * 100
                topo_acc = topo_acc / PRINT_ITER * 100
                assm_acc = assm_acc / PRINT_ITER * 100
                steo_acc = steo_acc / PRINT_ITER * 100

                print "KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f" % (kl_div, word_acc, topo_acc, assm_acc, steo_acc)
                word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0
                sys.stdout.flush()

            if (it + 1) % 1500 == 0: #Fast annealing
                scheduler.step()
                print "learning rate: %.6f" % scheduler.get_lr()[0]
                torch.save(model.state_dict(),  "molvae/vae-model/model.iter-%d-%d" % (epoch, it + 1))
                beta = max(1.0, beta + anneal)

        scheduler.step()
        print "learning rate: %.6f" % scheduler.get_lr()[0]
        torch.save(model.state_dict(),  "molvae/vae-model/model.iter-" + str(epoch))


#sample(100,450,56,3,'cpu','molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4')
#reconstruct(450,56,3,'cpu','molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4')
#pretrain_vae(450,56,3,40,'cpu')
train_vae(450,56,3, 0.005, 0, 0.0007, 40, 'cpu')
print "done"