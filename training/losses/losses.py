import torch
import torch.nn as nn
from losses.get_weights_from_training_data import gender_weights, race_weights, intersec_weights, gender_weights_subgroup, race_weights_subgroup, intersec_weights_subgroup, gender_weights_modified, race_weights_modified, intersec_weights_modified

class WeightedSampleCrossEntropyLoss(nn.Module):
    def __init__(self, attribute, weight_mode=None):
        super(WeightedSampleCrossEntropyLoss, self).__init__()
        if weight_mode:
            assert weight_mode in ['subgroup','prior_equal_dist'], f'The weight mode {weight_mode} is not supported, should be "subgroup" or "prior_equal_dist".'
        if not weight_mode:
            if attribute == 'gender':
                # self.weights = {'male': [1,1], # the first element is weight for real samples, the last element is weight for fake samples
                #                 'female': [1,1]}
                self.weights = gender_weights
            elif attribute == 'race':
                self.weights = race_weights
            elif attribute == 'intersec':
                self.weights = intersec_weights
        elif weight_mode == 'prior_equal_dist':
            if attribute == 'gender':
                # self.weights = {'male': [1,1], # the first element is weight for real samples, the last element is weight for fake samples
                #                 'female': [1,1]}
                self.weights = gender_weights_modified
            elif attribute == 'race':
                self.weights = race_weights_modified
            elif attribute == 'intersec':
                self.weights = intersec_weights_modified
        elif weight_mode == 'subgroup':
            if attribute == 'gender':
                # self.weights = {'male': [1,1], # the first element is weight for real samples, the last element is weight for fake samples
                #                 'female': [1,1]}
                self.weights = gender_weights_subgroup
            elif attribute == 'race':
                self.weights = race_weights_subgroup
            elif attribute == 'intersec':
                self.weights = intersec_weights_subgroup



    def forward(self, logits, labels, attributes):
        device = labels.device
        w = torch.Tensor([self.weights[a] for a in attributes]).to(device)
        loss = torch.tensor(0,dtype=torch.float, device=device)
        for idx, label in enumerate(labels):
            # weight = w[idx][label]
            # l = (- torch.log(torch.exp(logits[idx][label]) / torch.sum(torch.exp(logits[idx]))))
            # loss += weight * l

            loss += w[idx][label] * (- torch.log(torch.exp(logits[idx][label]) / torch.sum(torch.exp(logits[idx]))))
        loss = loss / logits.shape[0]
        return loss


if __name__ == '__main__':
    logits = torch.rand((10, 2),dtype=torch.float)
    print(logits)
    print('\n\n')
    labels = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long)
    attributes = ['male', 'female', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'female']
    criterion = nn.CrossEntropyLoss()
    mycriterion = WeightedSampleCrossEntropyLoss('gender')

    loss = criterion(logits,labels)
    myloss = mycriterion(logits,labels,attributes)

    print(gender_weights)
    print(race_weights)
    print(intersec_weights)

    print(f'nn.CrossEntropyLoss: {loss}.')
    print(f'my CrossEntropyLoss (all weights are 1): {myloss}.')
    print(f'Whether two losses equal: {loss == myloss}.')