# models/convlstm.py
import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )

class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False, output_steps=5):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.output_steps = output_steps  

        cell_list = []
        attention_convs = []
        residual_convs = []

        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))

            attention_convs.append(nn.Conv2d(self.hidden_dim[i], 1, kernel_size=1))
            if i > 0:
                residual_convs.append(nn.Conv2d(self.hidden_dim[i - 1], self.hidden_dim[i], kernel_size=1))

        self.cell_list = nn.ModuleList(cell_list)
        self.attention_convs = nn.ModuleList(attention_convs)
        self.residual_convs = nn.ModuleList(residual_convs)


        self.output_conv = nn.Sequential(
            nn.Conv3d(in_channels=self.hidden_dim[-1], out_channels=32, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=1, kernel_size=(1, 1, 1))
        )

    def forward(self, input_tensor, hidden_state=None):
        
        if not self.batch_first:
            
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        batch_size, seq_len, _, height, width = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError("Stateful ConvLSTM not implemented.")
        else:
            hidden_state = self._init_hidden(batch_size, (height, width))

        layer_output_list = []
        last_state_list = []

        current_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=current_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1) 

            B, T, C, H, W = layer_output.size()  
            
            layer_output_reshaped = layer_output.contiguous().view(B * T, C, H, W)
            
            attention_weights = self.attention_convs[layer_idx](layer_output_reshaped)
            attention_weights = torch.sigmoid(attention_weights)
           
            attention_weights = attention_weights.view(B, T, 1, H, W)
           
            layer_output = layer_output * attention_weights
            
          
            if layer_idx > 0:
                residual = self.residual_convs[layer_idx - 1](current_input.mean(dim=1))
                layer_output = layer_output + residual.unsqueeze(1)
            
            current_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        
        output = layer_output_list[-1] 
        output = output.permute(0, 2, 1, 3, 4)
        prediction = self.output_conv(output) 
        prediction = prediction[:, :, -self.output_steps:, :, :] 
        prediction = prediction.permute(0, 2, 1, 3, 4)

        return prediction

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
