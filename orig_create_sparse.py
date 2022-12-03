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
        pruning_method=prune.random_unstructured,
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

config = BertConfig.from_pretrained(
    'bert-base-uncased'
)
model = BertModel.from_pretrained(
            'bert-base-uncased',
            from_tf=bool(".ckpt" in 'bert-base-uncased'),
            config=config
        )
model.save_pretrained(f"full_prune/bert-base")

# sparsity_values = [0.1,0.2,0.3,0.4,0.45,0.49,0.495]
sparsity_values = [0.1]

for s in sparsity_values:
    model = BertModel.from_pretrained(
            'bert-base-uncased',
            from_tf=bool(".ckpt" in 'bert-base-uncased'),
            config=config
        )
    model = pruning_model_whole(model,s)
    model.save_pretrained(f"full_prune/bert-{s}")