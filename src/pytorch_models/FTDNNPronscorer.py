import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from src.pytorch_models.FTDNN import FTDNN

def summarize_outputs_per_phone(outputs, batch_target_phones, batch_indexes, normalize): 

    masked_outputs = outputs*abs(batch_target_phones)
    cuda0 = torch.device('cuda:0')
    
    by_phone_outputs_1 = torch.cat((
                      torch.zeros(masked_outputs.shape[0],1,masked_outputs.shape[2]).to(cuda0),
                      torch.cumsum(masked_outputs,1)[torch.tensor(batch_indexes[0,:]),torch.tensor(batch_indexes[1,:])]),  dim=1)
    by_phone_outputs_2 = torch.diff(by_phone_outputs_1,dim=1).to(cuda0)
    torch.save(by_phone_outputs_2, 'pouts.pt')
    if normalize:
    
        # Divides each instance of a phone in each batch by its duration in frames
        by_phone_frame_counts_1 = torch.cat((
                      torch.zeros(batch_target_phones.shape[0], 1, batch_target_phones.shape[2]).to(cuda0),
                      torch.cumsum(batch_target_phones,1)[torch.tensor(batch_indexes[0,:]), torch.tensor(batch_indexes[1,:])]),  dim=1)
        by_phone_frame_counts_2 = torch.diff(by_phone_frame_counts_1,dim=1)
        by_phone_frame_counts_2[ by_phone_frame_counts_2==0]=1
        by_phone_outputs = torch.div(by_phone_outputs_2, by_phone_frame_counts_2)
        by_phone_outputs.to(cuda0)
    else: 
        by_phone_outputs = by_phone_outputs_2
        
    return by_phone_outputs

def forward_by_phone(outputs, batch_target_phones, batch_indexes, phone_durs): 
    
    masked_outputs    = outputs*abs(batch_target_phones)
    collapsed_outputs = torch.sum(masked_outputs, dim=2)
    by_phone_outputs_1  = torch.diff(torch.cat((torch.zeros(collapsed_outputs.shape[0],1), 
                                   torch.cumsum(collapsed_outputs, dim=1)[batch_indexes]), dim=1))
    by_phone_outputs = torch.nan_to_num(torch.div(by_phone_outputs_1, phone_durs))

    return by_phone_outputs


class OutputLayer(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn=False):

        super(OutputLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bn = use_bn

        if use_bn:
            self.bn = nn.BatchNorm1d(self.in_dim, affine=False)
        self.linear = nn.Linear(self.in_dim, self.out_dim, bias=True) 
        self.nl = nn.Sigmoid()

    def forward(self, x):
        if self.use_bn:
            x = x.transpose(1,2)
            x =self.bn(x).transpose(1,2)
        x = self.linear(x)
        return x

class FTDNNPronscorer(nn.Module):

    def __init__(self, out_dim=40, batchnorm=None, dropout_p=0, device_name='cpu'):

        super(FTDNNPronscorer, self).__init__()

        use_final_bn = False
        if batchnorm in ["final", "last", "firstlast"]:
            use_final_bn=True
        
        self.ftdnn        = FTDNN(batchnorm=batchnorm, dropout_p=dropout_p, device_name=device_name)
        self.output_layer = OutputLayer(256, out_dim, use_bn=use_final_bn)
        
    def forward(self, x, loss_per_phone, evaluation, batch_target_phones, batch_indexes, normalize, phone_durs=None):
        '''
        Input must be (batch_size, seq_len, in_dim)
        '''
        x = self.ftdnn(x)
        x = self.output_layer(x)

        if loss_per_phone: 
            x = summarize_outputs_per_phone(x, batch_target_phones, batch_indexes, normalize)
        
        if evaluation:
            x = forward_by_phone(x, batch_target_phones, batch_indexes, phone_durs)
    
        return x
