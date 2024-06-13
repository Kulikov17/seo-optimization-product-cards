
import torch
import torch.nn as nn
import copy
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from safetensors.torch import load_model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import os

from tokenizers import Tokenizer


class CategoryModel(nn.Module):
    def __init__(
        self,
        module_features,
        num_classes: int,
        module_out_size: int = 1280,
        dropout_prob: float = 0.3,
        embed_size: int = 1024,

    ):
        super().__init__()
        self.module_features = module_features
        self.module_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.embed = nn.Sequential(
            nn.Linear(in_features=module_out_size,
                      out_features=embed_size),
        )
        self.classification = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features=embed_size,
                      out_features=num_classes),
        )

    def forward(self, inputs):
        step_1 = self.module_features(inputs)
        step_2 = self.module_avgpool(step_1)

        # эмбединги для RNN
        embeddings = self.embed(torch.flatten(step_2, 1))

        # классификация
        logits = self.classification(embeddings)

        return logits, embeddings


# переделать классы и передачу категорий
def build_model(model_name, num_classes, root_dir='./data'):
    pretrain_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    model = CategoryModel(module_features=pretrain_model.features,
                          num_classes=num_classes)

    model_path_safetensors = f'{root_dir}/{model_name}/model.safetensors'
    model_path_bin = f'{root_dir}/{model_name}/pytorch_model.bin'

    if os.path.isfile(model_path_safetensors):
        load_model(model, model_path_safetensors)
    else:
        model.load_state_dict(torch.load(model_path_bin,
                                         map_location=torch.device('cpu')))

    return model


class LSTMBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        lstm_layers: int,
    ):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

    def forward(self, packed, h0, c0):
        lstm_out, _ = self.lstm(packed, (h0, c0))
        return lstm_out, _


class DecoderRNN(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        image_embed_dim,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        lstm_layers: int = 1,
        dropout_prob: float = 0.1,
        lstm_parallel_layers: int = 1,
    ):
        super().__init__()

        self.lstm_layers = lstm_layers
        self.lstm_parallel_layers = lstm_parallel_layers
        self.hidden_dim = hidden_dim
        self.tokenizer = tokenizer

        self.image_embed_to_h0 = torch.nn.Sequential(
            torch.nn.Linear(in_features=image_embed_dim, out_features=lstm_layers * hidden_dim),
            torch.nn.LeakyReLU(0.1),
        )
        self.image_embed_to_c0 = torch.nn.Sequential(
            torch.nn.Linear(in_features=image_embed_dim, out_features=lstm_layers * hidden_dim),
            torch.nn.LeakyReLU(0.1),
        )

        self.embed = torch.nn.Embedding(
            num_embeddings=self.tokenizer.get_vocab_size(),
            embedding_dim=embed_dim,
            padding_idx=tokenizer.get_vocab()["[PAD]"],
        )

        for i in range(self.lstm_parallel_layers):
            self.add_module(f'lstm_block{i + 1}', LSTMBlock(embed_dim=embed_dim,
                                                            hidden_dim=hidden_dim,
                                                            lstm_layers=lstm_layers))

        self.linear = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_prob),
            torch.nn.Linear(in_features=hidden_dim * lstm_parallel_layers, out_features=tokenizer.get_vocab_size()),
        )

    def forward(self, image_embeds, captions):
        batch_size = image_embeds.shape[0]
        h0 = self.image_embed_to_h0(image_embeds).reshape(batch_size, self.lstm_layers, self.hidden_dim).transpose(0, 1)
        c0 = self.image_embed_to_c0(image_embeds).reshape(batch_size, self.lstm_layers, self.hidden_dim).transpose(0, 1)

        embeds = self.embed(captions)
        lengths = (captions != 0).sum(dim=-1).cpu()

        packed = pack_padded_sequence(embeds, lengths=lengths, batch_first=True, enforce_sorted=False)

        for i in range(self.lstm_parallel_layers):
            lstm_out, _ = self.__getattr__(f"lstm_block{i + 1}")(packed, h0, c0)
            output, _ = pad_packed_sequence(lstm_out, batch_first=True)

            if i == 0:
                outputs = torch.cat([output], dim=-1)
            else:
                outputs = torch.cat([outputs, output], dim=-1)

        logits = self.linear(outputs)

        return logits

    @torch.no_grad()
    def caption_image_beam_search(
        self,
        image_embeds: torch.Tensor,
        beam_size: int,
        max_length: int = 300,
    ):
        self.eval()

        batch_size = image_embeds.shape[0]
        if batch_size > 1:
            raise ValueError(f'Expected input batch_size (1) but got ({batch_size})')

        h = self.image_embed_to_h0(image_embeds).reshape(batch_size,
                                                         self.lstm_layers,
                                                         self.hidden_dim).transpose(0, 1)
        c = self.image_embed_to_c0(image_embeds).reshape(batch_size,
                                                         self.lstm_layers,
                                                         self.hidden_dim).transpose(0, 1)

        k = beam_size
        k_prev_words = torch.full((k, 1), self.tokenizer.get_vocab()["[BOS]"], device=image_embeds.device)
        seqs = k_prev_words

        top_k_scores = torch.zeros(k, 1, device=image_embeds.device)

        complete_seqs = list()
        complete_seqs_scores = list()
        uncomplete_seqs = list()
        uncomplete_seqs_scores = list()

        for i in range(self.lstm_parallel_layers):
            locals()[f'h{i+1}'] = h.expand(self.lstm_layers, k, self.hidden_dim)
            locals()[f'c{i+1}'] = c.expand(self.lstm_layers, k, self.hidden_dim)
            locals()[f'h{i+1}'], locals()[f'c{i+1}'] = eval(f"h{i+1}").contiguous(), eval(f"c{i+1}").contiguous()

        step = 1

        while True:
            embeds = self.embed(k_prev_words)
            for i in range(self.lstm_parallel_layers):
                output, (locals()[f"h{i+1}"], locals()[f"c{i+1}"]) = self.__getattr__(f"lstm_block{i+1}")(embeds,
                                                                                    eval(f"h{i+1}"),
                                                                                    eval(f"c{i+1}"))
                if i == 0:
                    outputs = torch.cat([output], dim=-1)
                else:
                    outputs = torch.cat([outputs, output], dim=-1)
            logits = self.linear(outputs)
            scores = F.log_softmax(logits[:, -1:, :], dim=-1)
            scores = scores.reshape(k, self.tokenizer.get_vocab_size())
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
                seqs = torch.cat([seqs, top_k_words.unsqueeze(1)], dim=-1)
            else:
                # выбираем 9 наилучших (по 3 для каждого предсказания)
                top_kk_scores, top_kk_words = scores.topk(k, -1, True, True)

                # выбираем из 9 наилучших 3 наилучшие
                top_k_scores, top_k_idx_flat = top_kk_scores.view(-1).topk(k, 0, True, True)
                top_k_idx_col = top_k_idx_flat // k
                top_k_idx_row = top_k_idx_flat % k
                top_k_idx = torch.cat([top_k_idx_col.unsqueeze(1), top_k_idx_row.unsqueeze(1)], dim=-1) 

                # переформировываем соответствующе 3м наилучшим seqs 
                for i in range(top_k_idx.shape[0]):
                    num_str = top_k_idx[i][0]
                    top_k_word = top_kk_words[num_str][top_k_idx[i][1]].unsqueeze(0) 
                    seq = torch.cat([seqs[num_str], top_k_word], dim=0).unsqueeze(0)

                    if i == 0:    
                        seqs_new = torch.cat([seq], dim=0)                        
                        for i in range(self.lstm_parallel_layers):
                            locals()[f'h_new{i+1}'] = torch.cat([eval(f"h{i+1}")[:, [num_str], :]], dim=1)
                            locals()[f'c_new{i+1}'] = torch.cat([eval(f"c{i+1}")[:, [num_str], :]], dim=1)

                        num_strs = torch.cat([num_str.unsqueeze(0)], dim=0)
                        top_k_words = torch.cat([top_k_word], dim=0)
                    else:
                        seqs_new = torch.cat([seqs_new, seq], dim=0)                      
                        for i in range(self.lstm_parallel_layers):
                            locals()[f'h_new{i+1}'] = torch.cat([eval(f"h_new{i+1}"), eval(f"h{i+1}")[:, [num_str], :]], dim=1)
                            locals()[f'c_new{i+1}'] = torch.cat([eval(f"c_new{i+1}"), eval(f"c{i+1}")[:, [num_str], :]], dim=1)

                        num_strs = torch.cat([num_strs, num_str.unsqueeze(0)], dim=0)
                        top_k_words = torch.cat([top_k_words, top_k_word], dim=0)
                seqs = copy.deepcopy(seqs_new)
                for i in range(self.lstm_parallel_layers):
                    locals()[f'h{i+1}'] = copy.deepcopy(eval(f"h_new{i+1}"))
                    locals()[f'c{i+1}'] = copy.deepcopy(eval(f"c_new{i+1}"))
            
            incomplete_inds = [ind for ind, next_word in enumerate(top_k_words) if
                               next_word != self.tokenizer.get_vocab()["[EOS]"]]
            complete_inds = list(set(range(len(top_k_words))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.append(seqs[complete_inds])
                complete_seqs_scores.append(top_k_scores[complete_inds])

            k -= len(complete_inds)        
            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            for i in range(self.lstm_parallel_layers):
                locals()[f'h{i+1}'] = eval(f"h{i+1}")[:, incomplete_inds, :].reshape(self.lstm_layers, k, self.hidden_dim)
                locals()[f'c{i+1}'] = eval(f"c{i+1}")[:, incomplete_inds, :].reshape(self.lstm_layers, k, self.hidden_dim)
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = seqs[:, -1:]                                  # seqs[:, -k:] сколько последних слов подавать в модель
            if step > max_length:
                uncomplete_seqs.append(seqs[incomplete_inds])
                uncomplete_seqs_scores.append(top_k_scores[incomplete_inds])
                break
            step += 1

        return complete_seqs, complete_seqs_scores, uncomplete_seqs, uncomplete_seqs_scores


def build_rnn_model(model_name='model_rnnv2', root_dir='./data/rnn_models'):
    tokenizer = Tokenizer.from_file(f"{root_dir}/tokenizer.json")
    model = DecoderRNN(
        tokenizer=tokenizer,
        image_embed_dim=1024,
        lstm_layers=2,
        lstm_parallel_layers=2,
        hidden_dim=256,
    )

    model.load_state_dict(torch.load(f"{root_dir}/{model_name}.pth", map_location=torch.device('cpu')))

    return model, tokenizer
