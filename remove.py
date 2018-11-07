import torch

state_dict = torch.load('inception_v3_google-1a9a5a14.pth')
new_state_dict = {key: state_dict[key] for key in state_dict if 'AuxLogits' not in key}
print(len(state_dict))
print(len(new_state_dict))
torch.save(new_state_dict, 'inception_v3_no_aux_logits.pth')
