import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import torch.nn.utils.rnn as rnn_utils

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        raise NotImplementedError

class TransPolicy(nn.Module):
    def __init__(self, vocab, max_len, n_heads, n_embed, n_layers, dropout):
        super(TransPolicy, self).__init__()

        self.vocab = vocab
        self.max_len = max_len        
        self.n_heads = n_heads
        self.n_embed = n_embed
        self._dropout = dropout
        self.n_layers = n_layers

        self.embedding = nn.Embedding(len(vocab), n_embed, padding_idx=vocab.pad, dtype=torch.float32)
        self.position_embedding = nn.Embedding(max_len, n_embed, dtype=torch.float32)
                
        encoder_layer = nn.TransformerEncoderLayer(n_embed, n_heads, 4 * n_embed, dropout=dropout, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers, nn.LayerNorm(n_embed))
        self.linear = nn.Linear(n_embed, len(vocab)) 
        weight_init(self.linear)

        self.register_buffer('triu', torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))

    def forward(self, x):
        L, B = x.shape
        x_tok = self.embedding(x)  #L, B
        x_pos = self.position_embedding(torch.arange(L, device=x.device)).view(L, 1, -1).expand(-1, B, -1)
        x = x_tok + x_pos
        x = self.encoder(x, self.triu[:L, :L])
        logits = self.linear(x)
        return td.Categorical(logits=logits)

    def autoregress(self, x):
        L, B = x.shape
        x_tok = self.embedding(x)  #L, B
        x_pos = self.position_embedding(torch.arange(L, device=x.device)).view(L, 1, -1).expand(-1, B, -1)
        x = x_tok + x_pos
        x = self.encoder(x, self.triu[:L, :L])
        logits = self.linear(x)[-1]
        return td.Categorical(logits=logits)

    def sample(self, batch_size, device, max_length):
        assert max_length <= self.max_len
        preds = self.vocab.bos * torch.ones((1, batch_size), dtype=torch.long, device=device)
        finished = torch.zeros((batch_size), dtype=torch.bool, device=device)
        imag_smiles_lens = torch.ones((batch_size),  device=device)

        with torch.no_grad():
            for i in range(1, max_length + 1):
                preds_dist = self.forward(preds)
                next_preds = preds_dist.sample()[-1].view(1, -1)
                preds = torch.cat([preds, next_preds], dim=0)
                imag_smiles_lens += ~finished

                EOS_sampled = (preds[-1] == self.vocab.eos)
                finished = torch.ge(finished + EOS_sampled, 1)
                if torch.prod(finished) == 1: break
        
        imag_smiles = preds.T.tolist()     
        return imag_smiles, imag_smiles_lens[0].tolist()

    def get_likelihood(self, obs, nonterms):      
        dist = self.forward(obs[:-1])
        logprobs = dist.log_prob(obs[1:]) * nonterms[:-1]
        return logprobs

    def get_data(self, batch_size, max_length, device):
        if max_length is None:
            max_length = self.max_len
        else:
            assert max_length <= self.max_len
            
        obs = torch.zeros((max_length + 1, batch_size), dtype=torch.long, device=device)
        obs[0] = self.vocab.bos
        nonterms = torch.zeros((max_length + 1, batch_size), dtype=torch.bool, device=device)
        rewards = torch.zeros((max_length, batch_size), dtype=torch.float32, device=device)
        end_flags = torch.zeros((1, batch_size), dtype=torch.bool, device=device)

        for i in range(1, max_length+1):
            preds_dist = self.autoregress(obs[:i])
            preds = preds_dist.sample()
 
            obs[i] = preds
            nonterms[i-1] = ~end_flags
            
            EOS_sampled = (preds == self.vocab.eos)
            rewards[i-1] = EOS_sampled * (~end_flags)

            #check if all sequences are done
            end_flags = torch.ge(end_flags + EOS_sampled, 1)
            if torch.prod(end_flags) == 1: break

        if i == max_length:
            rewards[-1] = rewards[-1] + (~end_flags)

        #remove assertion afterwards
        assert rewards.sum() == batch_size

        obs = obs[:i+1]
        nonterms = nonterms[:i+1]
        rewards = rewards[:i]
        episode_lens = nonterms.sum(0).cpu()

        return obs, rewards, nonterms, episode_lens

    def get_save_dict(self):
        return {
            "embedding": self.embedding.state_dict(),
            "position_embedding": self.position_embedding.state_dict(),
            "encoder": self.encoder.state_dict(),
            "linear": self.linear.state_dict(),
        }

    def load_save_dict(self, saved_dict):
        self.embedding.load_state_dict(saved_dict["embedding"])
        self.position_embedding.load_state_dict(saved_dict["position_embedding"])
        self.encoder.load_state_dict(saved_dict["encoder"])
        self.linear.load_state_dict(saved_dict["linear"])

class RnnPolicy(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, num_layers):
        super(RnnPolicy, self).__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), embedding_size, padding_idx=vocab.pad, dtype=torch.float32)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, len(vocab))
    
    def forward(self, x, lengths, hiddens=None):
        x = self.embedding(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, enforce_sorted=False)
        x, hiddens = self.rnn(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x)
        logits = self.linear(x)
        return td.Categorical(logits=logits), lengths, hiddens

    def sample(self, batch_size, max_length, device):
        starts = self.vocab.bos * torch.ones((1, batch_size), dtype=torch.long, device=device)
        finished = torch.zeros((1, batch_size), dtype=torch.bool, device=device)

        imag_smiles = [starts]
        imag_smiles_lens = torch.ones((1, batch_size),  device=device)
        
        input_lens = torch.ones(batch_size)
        hiddens = None
        with torch.no_grad():
            for i in range(1, max_length + 1):
                preds_dist, _, hiddens = self.forward(starts, input_lens, hiddens)
                preds = preds_dist.sample()
                imag_smiles.append(preds)
                imag_smiles_lens += ~finished
                starts = preds

                EOS_sampled = (preds == self.vocab.eos)
                finished = torch.ge(finished + EOS_sampled, 1)
                if torch.prod(finished) == 1: break

        imag_smiles = torch.cat(imag_smiles, 0).T.tolist()
        return imag_smiles, imag_smiles_lens[0].tolist()
    
    def get_likelihood(self, obs, smiles_len, nonterms):      
        dist, _, _ = self.forward(obs[:-1], smiles_len)
        logprobs = dist.log_prob(obs[1:]) * nonterms[:-1]
        return logprobs

    def get_data(self, batch_size, max_length, device):
        obs = torch.zeros((max_length + 1, batch_size), dtype=torch.long, device=device)
        obs[0] = self.vocab.bos
        nonterms = torch.zeros((max_length + 1, batch_size), dtype=torch.bool, device=device)
        rewards = torch.zeros((max_length, batch_size), dtype=torch.float32, device=device)
        end_flags = torch.zeros((1, batch_size), dtype=torch.bool, device=device)

        input_lens = torch.ones(batch_size)
        hiddens = None
        for i in range(1, max_length+1):
            preds_dist, _, hiddens = self.forward(obs[i-1].view(1, -1), input_lens, hiddens)
            preds = preds_dist.sample()

            obs[i] = preds
            nonterms[i-1] = ~end_flags
            
            EOS_sampled = (preds == self.vocab.eos)
            rewards[i-1] = EOS_sampled * (~end_flags)

            #check if all sequences are done
            end_flags = torch.ge(end_flags + EOS_sampled, 1)
            if torch.prod(end_flags) == 1: break

        if i == max_length:
            rewards[-1] = rewards[-1] + (~end_flags)

        #remove assertion afterwards
        assert rewards.sum() == batch_size

        obs = obs[:i+1]
        nonterms = nonterms[:i+1]
        rewards = rewards[:i]
        episode_lens = nonterms.sum(0).cpu()

        return obs, rewards, nonterms, episode_lens
    
    def get_save_dict(self):
        return {
            "embedding": self.embedding.state_dict(),
            "rnn": self.rnn.state_dict(),
            "linear": self.linear.state_dict(),
        }

    def load_save_dict(self, saved_dict):
        self.embedding.load_state_dict(saved_dict["embedding"])
        self.rnn.load_state_dict(saved_dict["rnn"])
        self.linear.load_state_dict(saved_dict["linear"])