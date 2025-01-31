import torch
import torch.nn as nn

class MaskedLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, input):
        res = []
        if type(self.mask) == list:
            outs = self.base(input)
            for i in range(len(outs)):
                res.append(self.mask[i] * outs[i])
            return res
        else:
            return self.base(input) * self.mask

class MaskedRNNLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedRNNLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, input, hx):
        res = []
        outs = self.base(input, hx)
        num_out_ = len(outs) if type(outs) in [tuple, list] else 1
        if num_out_ > 1:
            for i in range(num_out_):
                res.append(self.mask[i] * outs[i])
            return res
        else:
            return outs * self.mask[0]

class MaskedGRULayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedGRULayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, input):
        res = []
        outs = self.base(input)
        res.append(self.mask[0] * outs[0])
        res.append(self.mask[1].unsqueeze(1) * outs[1])
        return res

class MaskedATTLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedATTLayer, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, query, key, value, mask=None, dropout=None):
        return self.base(query, key, value, mask, dropout) * self.mask

class MaskedTENCLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedTENCLayer, self).__init__()
        self.base = base
        self.mask = mask.unsqueeze(1)

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        return self.base(src, mask, src_key_padding_mask, is_causal) * self.mask

class MaskedPOSENCCLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedPOSENCCLayer, self).__init__()
        self.base = base
        self.mask = mask.unsqueeze(1).cpu()

    def forward(self, input):
        return self.base(input) * self.mask

class MaskedObservationPropagation(nn.Module):
    def __init__(self, base, mask):
        super(MaskedObservationPropagation, self).__init__()
        self.base = base
        self.mask = mask

    def forward(self, x, p_t, edge_index, edge_weights=None,
            use_beta=False, edge_attr=None, return_attention_weights=None):
        res = []
        outs = self.base(x, p_t, edge_index, edge_weights, use_beta, edge_attr, return_attention_weights)
        num_out_ = len(outs) if type(outs) in [tuple, list] else 1
        cnt = 0
        if num_out_ > 1:
            for i in range(num_out_):
                tmp = outs[i]
                if type(tmp) not in [list, tuple]:
                    res.append(outs[i] * self.mask[cnt])
                    cnt += 1
                else:
                    t_res = []
                    t_res.append(outs[i][0])
                    t_res.append(outs[i][1] * self.mask[cnt])
                    cnt += 1
                    res.append(t_res)
            return res
        else:
            return outs * self.mask[0]

class Pruning:
    def __init__(self,
                 training_loader=None,
                 model=None):
        self.training_loader = training_loader
        self.model = model

    def repair(self, model_type, device="cuda"):
        device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        model = self.model.model
        model.to(device)

        layers = []
        att_layers = dict()
        if model_type == "BRITS":
            ms = [getattr(model, "rits_b"), getattr(model, "rits_f")]
            atts = ["rnn_cell", "temp_decay_h", "temp_decay_x", "hist_reg", "feat_reg", "combining_weight"]
            for m in ms:
                for att in atts:
                    layers.append(getattr(m, att))
                    att_layers[layers[-1]] = (m, att)
        elif model_type == "GRUD":
            atts = ["rnn_cell", "temp_decay_h", "temp_decay_x"]
            for att in atts:
                layers.append(getattr(model, att))
                att_layers[layers[-1]] = (model, att)
        elif model_type == "Raindrop":
            layers.append(getattr(model, "transformer_encoder"))
            att_layers[layers[-1]] = (model, "transformer_encoder")
            atts = ["pos_encoder"]
            for att in atts:
                layers.append(getattr(model, att))
                att_layers[layers[-1]] = (model, att)
        elif model_type == "iTransformer":
            atts = ["embedding_layer", "encoder", "output_projection", "saits_embedding"]
            for att in atts:
                layers.append(getattr(model, att))
                att_layers[layers[-1]] = (model, att)

        containers, hooks = dict(), []
        for layer in layers:
            containers[layer] = []
            def forward_hook(module, input, output):
                containers[module].append(output)
            hooks.append(layer.register_forward_hook(forward_hook))
        with torch.no_grad():
            print("Forwarding all training set")
            model.eval()
            for idx, data in enumerate(self.training_loader):
                inputs = self.model._assemble_input_for_training(data)
                model.forward(inputs)
            for hook in hooks: hook.remove()
        return layers, containers, att_layers

    def prune(self, layers, containers, att_layers, prune_rate):
        for layer in layers:
            out = containers[layer]
            (model, layer_to_prune) = att_layers[layer]
            if layer_to_prune == "rnn_cell":
                num_out_, masks = len(out[0]) if type(out[0]) in [tuple, list] else 1, []
                for i in range(num_out_):
                    container = torch.cat(out, dim=0) if num_out_ == 1 else torch.cat([o[i] for o in out], dim=0)
                    activation = torch.mean(container, dim=[0])
                    out_shape = activation.shape
                    activation = activation.reshape(-1)
                    seq_sort = torch.argsort(activation)
                    prunned_neurons, mask = (int(activation.shape[0] * prune_rate), torch.ones(activation.shape).cuda())
                    mask[seq_sort[:prunned_neurons]] = 0
                    masks.append(mask.reshape(out_shape))
                setattr(model, layer_to_prune, MaskedRNNLayer(layer, masks))
            elif layer_to_prune == "gru_rnn":
                num_out_, masks = len(out[0]) if type(out[0]) in [tuple, list] else 1, []
                for i in range(num_out_):
                    if i == 0:
                        container = torch.cat([o[i] for o in out], dim=0)
                        activation = torch.mean(container, dim=[0])
                    else:
                        container = torch.cat([o[i] for o in out], dim=1)
                        activation = torch.mean(container, dim=[1])
                    out_shape = activation.shape
                    activation = activation.reshape(-1)
                    seq_sort = torch.argsort(activation)
                    prunned_neurons, mask = (int(activation.shape[0] * prune_rate), torch.ones(activation.shape).cuda())
                    mask[seq_sort[:prunned_neurons]] = 0
                    masks.append(mask.reshape(out_shape))
                setattr(model, layer_to_prune, MaskedGRULayer(layer, masks))
            elif layer_to_prune in ["ob_propagation", "ob_propagation_layer2"]:
                num_out_, masks = len(out[0]) if type(out[0]) in [tuple, list] else 1, []
                for i in range(num_out_):
                    if num_out_ == 1:
                        container = torch.cat(out, dim=0)
                    else:
                        tmp = [o[i] for o in out]
                        if type(tmp[0]) not in [list, tuple]:
                            container = torch.cat(tmp, dim=0)
                        else:
                            container = torch.cat([tp[1] for tp in tmp], dim=0)

                    activation = torch.mean(container, dim=[0])
                    out_shape = activation.shape
                    activation = activation.reshape(-1)
                    seq_sort = torch.argsort(activation)
                    prunned_neurons, mask = (int(activation.shape[0] * prune_rate), torch.ones(activation.shape).cuda())
                    mask[seq_sort[:prunned_neurons]] = 0
                    masks.append(mask.reshape(out_shape))
                setattr(model, layer_to_prune, MaskedObservationPropagation(layer, masks))
            else:
                if layer_to_prune in ["pos_encoder", "transformer_encoder"]:
                    container = torch.cat(containers[layer], dim=1)
                    # activation = torch.mean(container, dim=[0, 1])
                    activation = torch.mean(container, dim=[1])
                else:
                    container = torch.cat(containers[layer], dim=0)
                    activation = torch.mean(container, dim=[0])
                out_shape = activation.shape
                activation = activation.reshape(-1)
                seq_sort = torch.argsort(activation)
                prunned_neurons, mask = int(activation.shape[0] * prune_rate), torch.ones(activation.shape).cuda()
                mask[seq_sort[:prunned_neurons]] = 0
                if layer_to_prune == "att":
                    setattr(model, layer_to_prune, MaskedATTLayer(layer, mask.reshape(out_shape)))
                elif layer_to_prune == "transformer_encoder":
                    setattr(model, layer_to_prune, MaskedTENCLayer(layer, mask.reshape(out_shape)))
                elif layer_to_prune == "pos_encoder":
                    setattr(model, layer_to_prune, MaskedPOSENCCLayer(layer, mask.reshape(out_shape)))
                else:
                    setattr(model, layer_to_prune, MaskedLayer(layer, mask.reshape(out_shape)))

