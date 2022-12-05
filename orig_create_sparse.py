import matplotlib.pyplot as plt
import torch
import torch.nn.utils.prune as prune
import numpy as np
from transformers import BertModel
from transformers import BertConfig

def pruning_model_whole(model,px):

    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.query, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.key, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.value, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.output.dense, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].output.dense, 'weight'))

    parameters_to_prune.append((model.pooler.dense, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )
    for (layer,wt) in parameters_to_prune:
        print('in remove')
        print(list(layer.named_buffers()))
        prune.remove(layer, wt)
    
    return model

def pruning_model_random(model,px):

    parameters_to_prune =[]
    for ii in range(12):
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.query, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.key, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.value, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.output.dense, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].output.dense, 'weight'))

    parameters_to_prune.append((model.pooler.dense, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )
    for (layer,wt) in parameters_to_prune:
        print('in remove')
        print(list(layer.named_buffers()))
        prune.remove(layer, wt)
    
    return model

def pruning_model_upper(model,px):

    parameters_to_prune =[]
    for ii in range(6):
        print(ii)
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.query, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.key, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.value, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.output.dense, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].output.dense, 'weight'))

    parameters_to_prune.append((model.pooler.dense, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=2*px,
    )
    
    for (layer,wt) in parameters_to_prune:
        print('in remove')
        print(list(layer.named_buffers()))
        prune.remove(layer, wt)
    
    return model

def pruning_model_lower(model,px):

    parameters_to_prune =[]
    for i in range(6):
        ii = i + 6
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.query, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.key, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.self.value, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].attention.output.dense, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].intermediate.dense, 'weight'))
        parameters_to_prune.append((model.encoder.layer[ii].output.dense, 'weight'))

    parameters_to_prune.append((model.pooler.dense, 'weight'))
    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=2*px,
    )
    for (layer,wt) in parameters_to_prune:
        print('in remove')
        print(list(layer.named_buffers()))
        prune.remove(layer, wt)
    
    return model

def see_weight_rate(model):

    sum_list = 0
    zero_sum = 0
    for ii in range(12):
        sum_list = sum_list+float(model.encoder.layer[ii].attention.self.query.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.encoder.layer[ii].attention.self.query.weight == 0))

        sum_list = sum_list+float(model.encoder.layer[ii].attention.self.key.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.encoder.layer[ii].attention.self.key.weight == 0))

        sum_list = sum_list+float(model.encoder.layer[ii].attention.self.value.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.encoder.layer[ii].attention.self.value.weight == 0))

        sum_list = sum_list+float(model.encoder.layer[ii].attention.output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.encoder.layer[ii].attention.output.dense.weight == 0))

        sum_list = sum_list+float(model.encoder.layer[ii].intermediate.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.encoder.layer[ii].intermediate.dense.weight == 0))

        sum_list = sum_list+float(model.encoder.layer[ii].output.dense.weight.nelement())
        zero_sum = zero_sum+float(torch.sum(model.encoder.layer[ii].output.dense.weight == 0))


    sum_list = sum_list+float(model.pooler.dense.weight.nelement())
    zero_sum = zero_sum+float(torch.sum(model.pooler.dense.weight == 0))
 

    return 100*zero_sum/sum_list


config = BertConfig.from_pretrained(
    'bert-base-uncased'
)
# model = BertModel.from_pretrained(
#             'bert-base-uncased',
#             from_tf=bool(".ckpt" in 'bert-base-uncased'),
#             config=config
#         )
# model.save_pretrained(f"full_prune/bert-base")

sparsity_values = [0.7,0.9,0.95,0.99]
# sparsity_values = [0.495]


for s in sparsity_values:
    model = BertModel.from_pretrained(
            'bert-base-uncased',
            from_tf=bool(".ckpt" in 'bert-base-uncased'),
            config=config
        )
    model = pruning_model_random(model,s)

    zero = see_weight_rate(model)
    print('zero rate', zero)

    model.save_pretrained(f"upper_prune/bert-{s}")