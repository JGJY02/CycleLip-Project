from os import kill
from tkinter import Frame
from turtle import forward
import torch
from torch import nn
from math import sqrt
from hparams import hparams as hps
from torch.autograd import Variable
from torch.nn import functional as F
from .lip2wav_layers import Conv3d, LinearNorm, ConvolutionEncoder, custom_conv1d, HighwayNet
from utils.util import mode, get_mask_from_lengths

#For testing purposes
import sys


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction = 'none')

    def forward(self, model_outputs, targets, trainable_params):
        mel_out, mel_out_postnet, _ , output_lengths = model_outputs
        # gate_out = gate_out.view(-1, 1)
        # print(mel_out == mel_out_postnet)

        mel_target, gate_target, output_lengths = targets
        mel_target.requires_grad = False
        # gate_target.requires_grad = False
        output_lengths.requires_grad = False
        slice = torch.arange(0, gate_target.size(1), hps.outputs_per_step)
        # gate_target = gate_target[:, slice].view(-1, 1)
        mel_mask = ~get_mask_from_lengths(output_lengths.data) #Removed True from masking as no padding required JAred

        # mel_loss = self.loss(mel_out, mel_target) + \
        #     self.loss(mel_out_postnet, mel_target)
        # mel_loss = mel_loss.sum(1).masked_fill_(mel_mask, 0.)/mel_loss.size(1)
        # mel_loss = mel_loss.sum()/output_lengths.sum()

        MSE_Loss = nn.MSELoss()
        L1_Loss = nn.L1Loss()
        
        

        # MSE Loss
        before = MSE_Loss(mel_out, mel_target)
        after = MSE_Loss(mel_out_postnet, mel_target)
        # L1 Loss
        l1 = L1_Loss(mel_out, mel_target)
        # Regularization
        if hps.tacotron_scale_regularization:
            reg_weight_scaler = 1. / (
                        2 * hps.max_abs_value) if hps.symmetric_mels else 1. / (
                hps.max_abs_value)
            reg_weight = hps.tacotron_reg_weight * reg_weight_scaler
        else:
            reg_weight = hps.tacotron_reg_weight

        
            regularization = torch.sum(torch.tensor([torch.norm(v, p=2) for v in trainable_params]) )* reg_weight

        Total_loss = before + after + l1 + regularization

        # gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return Total_loss, (before.item(), after.item(), l1.item(), regularization.item())


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        # "location_features_convolution"
        self.location_conv = torch.nn.Conv1d(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        
        # "location_features_layer"
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')


    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, decoder_lstm_units, embedding_dim, attention_dim,
                 attention_filters, attention_kernel):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(decoder_lstm_units, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm((embedding_dim+hps.speaker_embedding_size), attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_filters,
                                            attention_kernel,
                                            attention_dim)
        self.score_mask_value = -float('inf')
        self.attention_variable_projection = nn.Parameter(torch.empty(1, hps.attention_dim).cuda())
        torch.nn.init.xavier_uniform_(self.attention_variable_projection)

        self.attention_bias = nn.Parameter(torch.zeros(1, hps.attention_dim).cuda())



    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        '''
        PARAMS
        ------
        query: decoder output (batch, num_mels * outputs_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        '''

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        # print(attention_weights_cat.shape)

        #Added new by jared for attention variable projection and attention bias
        #dtype = query.dtype
        # num_units = processed_attention_weights.shape[-1]
        # print(num_units)


        # print("self.attention_variable_projection shape {}".format(self.attention_variable_projection.shape))
        # print("self.attention_bias shape {}".format(self.attention_bias.shape))
        # print("processed_memory shape {}".format(processed_memory.shape))
        # print("attention_weights_cat shape {}".format(processed_attention_weights.shape))
        # print("processed_query shape {}".format(processed_query.shape))
        # end of addition

        energies = self.v(self.attention_variable_projection * torch.tanh(
            processed_query + processed_attention_weights  + processed_memory + self.attention_bias))
        
        energies = energies.squeeze(-1)
        # print(energies.shape)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        '''
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        '''
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)
        # print("Alignment shape {}".format(alignment.shape))
        ## No masking anymore Jared
        # if mask is not None:
        #     alignment.data.masked_fill_(mask, self.score_mask_value)
        


        attention_weights = F.softmax(alignment, dim=1)
        # print("attention_weights shape {}".format(attention_weights.shape))
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        # print("memory at attention context shape {}".format(memory.shape))
        # print("attention_context shape {}".format(attention_context.shape))
        attention_context = attention_context.squeeze(1)



        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        # print("Input size for prenet is {}".format(x.shape))
        i = 0
        for linear in self.layers:
            i += 1
            # print("This is the input shape of prenet layer {}, {}".format(i, x.shape))
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    '''Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    '''

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.linear_projection = LinearNorm(hps.postnet_channels,
            hps.num_mels
            )
        self.activation = torch.nn.Tanh()
 #No batch normalisations for postnet in lip2wav
        self.convolutions.append(
            nn.Sequential(
                custom_conv1d(hps.num_mels, hps.postnet_channels, activation = self.activation,
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
            )
        )

        for i in range(1, hps.postnet_num_layers - 1):
            self.convolutions.append(
                nn.Sequential(
                    custom_conv1d(hps.postnet_channels,
                             hps.postnet_channels, activation = self.activation,
                             kernel_size=hps.postnet_kernel_size, stride=1,
                             padding=int((hps.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    )
                )
            

        self.convolutions.append(
            nn.Sequential(
                custom_conv1d(hps.postnet_channels, hps.postnet_channels, activation = torch.nn.Identity(),
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear')
                # nn.BatchNorm1d(hps.num_mels))
            )
            )

    def forward(self, x):
        
        for i in range(len(self.convolutions) - 1):
            #print("Current Postnet x shape is {}".format(x.shape))
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        # num_of_params_conv = sum(p.numel() for p in self.convolutions.parameters())/1000000
        # print("postnet projection parameters {} million".format(num_of_params_conv))
        # print("The output shape of postnet is {}".format(x.shape))
        # print("Convolution parameters {} million".format(sum(p.numel() for p in self.convolutions[i].parameters()))/1000000)
        x = x.transpose(1,2)
        x = self.linear_projection(x) #Forward Jared postnet addition so that we do projection
        # # num_of_params_proj = sum(p.numel() for p in self.linear_projection.parameters())/1000000
        # # print("postnet projection parameters {} million".format(num_of_params_proj))
        # x = x.reshape(hps.tacotron_batch_size, hps.num_mels, hps.num_mels)
        # print("Forward Projection shape {}".format(x.shape))
        return x


class Encoder(nn.Module):
    '''Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    '''
    def __init__(self):
        super(Encoder, self).__init__()

        convolutions = []
        # for i in range(hps.encoder_n_convolutions):
        #     conv_layer = nn.Sequential(
        #         Conv3d(hps.symbols_embedding_dim if i == 0 \
        #                     else hps.embedding_dim,
        #                  hps.embedding_dim,
        #                  kernel_size=hps.encoder_kernel_size, stride=1,
        #                  padding=int((hps.encoder_kernel_size - 1) / 2),
        #                  dilation=1, w_init_gain='relu'),
        #         nn.BatchNorm1d(hps.embedding_dim))
        #     convolutions.append(conv_layer)
        # self.convolutions = nn.ModuleList(convolutions)

        # 'encoder_convolutions'
        self.convolutions = ConvolutionEncoder()

        # 'encoder_LSTM'
        self.lstm = nn.LSTM(hps.embedding_dim,
                            hps.encoder_lstm_units, 1,
                            batch_first=True, bidirectional=True)


    def forward(self, x, input_lengths):
        cnt = 0
        #print(x.shape)
        #print(torch.cuda.memory_allocated()/ (1024**3))
        # print(x.shape)
        for conv in self.convolutions.Convolution_Encoder_Blocks:
            # x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            # print("Currently on block {}".format(cnt))
            # print("Current input shape is {}".format(x.shape))
            # print("currently on block {}".format(cnt))
            x = conv(x)
            cnt += 1
        # print("Output shape from convolution {}".format (x.shape))
        
        #print("Current shape after convolution is {}".format(x.shape))
        x_shape = x.shape[4]*x.shape[3]*x.shape[2]
        x = x.reshape(-1, x.shape[1], x_shape)
        
        # print("The current shape of the output is {}".format(x.shape))
        #We want the hidden dimension to be 512
        x = x.transpose(1, 2)
        # print("The lstm input shape is {}".format(x.shape))

        # pytorch tensor are not reversible, hence the conversion
        # print("input_lengths {}".format(input_lengths))
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()

        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        
        # print("Output shape from encoder {}".format (outputs.shape)) # Jared error checker
        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # "decoder_prenet"
        self.prenet = Prenet(
            hps.num_mels,
            [hps.prenet_dim, hps.prenet_dim])
        

        self.attention_rnn = nn.LSTMCell(
            hps.decoder_lstm_units,
            hps.decoder_lstm_units)
        

        self.attention_layer = Attention(
            hps.decoder_lstm_units, hps.embedding_dim,
            hps.attention_dim, hps.attention_filters,
            hps.attention_kernel)

        # "decoder_LSTM"
        self.decoder_rnn = nn.LSTMCell(
            hps.decoder_lstm_units ,
            hps.decoder_lstm_units, 2)
        

        self.linear_projection = LinearNorm(
            hps.decoder_lstm_units + hps.embedding_dim + hps.speaker_embedding_size,
            hps.num_mels * hps.outputs_per_step)

        ## Removed as this is the stop projection
        # self.gate_layer = LinearNorm(
        #     hps.decoder_lstm_units + hps.embedding_dim, 1,
        #     bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        ''' Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        '''
        B = memory.size(0)
        # decoder_input = Variable(memory.data.new(
        #     B, hps.num_mels * hps.outputs_per_step).zero_())
        decoder_input = Variable(memory.data.new(
            B, hps.num_mels).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        ''' Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        '''
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, hps.decoder_lstm_units).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, hps.decoder_lstm_units).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, hps.decoder_lstm_units).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, hps.decoder_lstm_units).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, hps.embedding_dim + hps.speaker_embedding_size).zero_())

        self.memory = memory
        # print("Memory shape at decoder state intialization {}".format(self.memory.shape))
        self.processed_memory = self.attention_layer.memory_layer(memory)
        # print("Processed memory shape {}".format(self.processed_memory.shape))
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        ''' Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        '''
        # # (B, num_mels, T_out) -> (B, T_out, num_mels)
        # # print("parse_decoder_inputs shape 1 {}".format(decoder_inputs.shape))
        # decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
        # decoder_inputs = decoder_inputs.view(
        #     decoder_inputs.size(0),
        #     int(decoder_inputs.size(1)/hps.outputs_per_step), -1)
        # # print("parse_decoder_inputs shape 2 {}".format(decoder_inputs.shape))
        # # (B, T_out, num_mels) -> (T_out, B, num_mels)
        # decoder_inputs = decoder_inputs.transpose(0, 1)
        # # print("parse_decoder_inputs shape 3 {}".format(decoder_inputs.shape))

        decoder_inputs = decoder_inputs[:, hps.outputs_per_step-1:: hps.outputs_per_step, :]
        decoder_inputs = decoder_inputs.transpose(0,1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, alignments):
        ''' Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        '''
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        # gate_outputs = torch.stack(gate_outputs).transpose(0, 1) Removing gate outputs Jared
        # gate_outputs = gate_outputs.contiguous()
        # (T_out, B, num_mels) -> (B, T_out, num_mels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        # print(mel_outputs.shape)
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, hps.num_mels)
        # (B, T_out, num_mels) -> (B, num_mels, T_out)
        # print(mel_outputs.shape)
        # mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, alignments

    def decode(self, decoder_input):
        ''' Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        '''
        
        # print("shape of decoder input is {}".format(decoder_input.shape))
        # print("shape of attention_context is {}".format(self.attention_context.shape))
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        
        
        # ## Temp removal uncomment if doesnt work Jared
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, hps.tacotron_zoneout_rate, self.training)
        
        # print("Attention hidden {}" .format(self.attention_hidden.shape))
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        # print("attention_weights_cat shape {}".format(attention_weights_cat.shape))
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (decoder_input, self.attention_context), -1)
        # print("decoder_input shape is {}".format(decoder_input.shape))
        # print("self.attention_context shape is {}".format(self.attention_context.shape))

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, hps.tacotron_zoneout_rate, self.training)

        # print("Decoder input shape {}".format(decoder_input.shape))
        
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        # print("Decoder output shape after projection {}".format(decoder_output.shape))

        # gate_prediction = self.gate_layer(decoder_hidden_attention_context) Removed gate prediction output
        return decoder_output, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        ''' Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        '''
        ## memory comes in as [Batch_size, lstm_output, Embedding_size + speaker_embedding_size]
        # print(memory.shape)
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        # print("Get_go_frame output {}".format(decoder_input.shape))
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        # print("shape of parse decoder inputs output is {}".format(decoder_inputs.shape))
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        # print("the output shape of the decoder input is {} whilst memory {}".format(decoder_inputs.shape, memory.shape))
        
        decoder_inputs = self.prenet(decoder_inputs)
        # print("the output shape of prenet is {}".format(decoder_inputs.shape))
        # print("Memory Length {}".format(memory_lengths))
        mask = get_mask_from_lengths(memory_lengths)
        self.initialize_decoder_states(
            memory, mask=~mask)

        # mel_outputs, gate_outputs, alignments = [], [], []
        # while len(mel_outputs) < decoder_inputs.size(0) - 1:
        #     decoder_input = decoder_inputs[len(mel_outputs)]
        #     #Remove gate output Jared
        #     mel_output, gate_output, attention_weights = self.decode(
        #         decoder_input)
        #     mel_outputs += [mel_output.squeeze(1)]
        #     gate_outputs += [gate_output.squeeze()]
        #     alignments += [attention_weights]
        # mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
        #     mel_outputs, gate_outputs, alignments)
    

        mel_outputs, alignments = [], []
        # print(decoder_inputs.size(0) - 1)
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            # print(decoder_input.shape)
            #Remove gate output Jared
            
            mel_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            alignments += [attention_weights]
        mel_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, alignments)
        
        # print("mel_outputs shape {}".format(mel_outputs.shape))
        return mel_outputs, alignments

    def inference(self, memory): # Removed gate outputs Jared
        ''' Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        '''
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, alignments = [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            # gate_outputs += [gate_output]
            alignments += [alignment]

            if hps.outputs_per_step*len(mel_outputs)/alignment.shape[1] \
                    >= hps.max_decoder_ratio:
                print('Warning: Reached max decoder steps.')
                break

            decoder_input = mel_output

        mel_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, alignments)
        return mel_outputs, alignments

# class CBHG:
#     def __init__(self):
#         super(CBHG, self).__init__()
#         self.process = nn.ModuleList()
#         self.max_pooling = nn.MaxPool1d(hps.cbhg_pool_size)
#         self.proj1_conv = Conv1d(hps.cbhg_pool_size, hps.cbhg_projection, hps.cbhg_projection_kernel_size)
#         self.proj2_conv = Conv1d(hps.cbhg_projection, hps.num_mels, hps.cbhg_projection_kernel_size)
        
#         self.highwaynet_layers = HighwayNet(hps.cbhg_highway_units, hps.cbhg_highway_units)
#         self.CBHG_bidrectional = nn.GRU(hps.cbhg_rnn_units,
#                             hps.cbhg_rnn_units, 1,
#                             batch_first=True, bidirectional=True)
#  #No batch normalisations for postnet in lip2wav
#     def forward(self, x):
#         for k in range(1, hps.cbhg_kernels + 1):
#             conv_block = nn.sequential( Conv1d(hps.cbhg_conv_channels, hps.cbhg_conv_channels, k), nn.ReLU())
#             conv_outputs = torch.concat(conv_block(x), axis=-1)
       
#         pooling_output = self.max_pooling(conv_outputs)
#         proj1_output = self.proj1_conv(pooling_output)
#         proj2_output = self.proj2_conv(proj1_output)

#         highway_input = proj2_output + x

#         if highway_input.shape[2] != hps.cbhg_highway_units:
#             self.highway_linear = LinearNorm(highway_input.shape[2], hps.cbhg_highway_units)
#             hgihway_input = self.highway_linear(highway_input)

#         # 4-layer HighwayNet
#         for highwaynet in self.highwaynet_layers:
#             highway_input = highwaynet(highway_input)
#         rnn_input = highway_input

#     # Bidirectional RNN
        
#         rnn_input_packed = nn.utils.rnn.pack_padded_sequence(rnn_input, None, batch_first=True)
#         self.lstm.flatten_parameters()
#         outputs, states = self.CBHG_bidrectional(rnn_input_packed)

#         return torch.concat(outputs, axis=2)  # Concat forward and backward outputs


        


class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()
        # No longer require word embeddings as we are using frames Jared
        # self.embedding = nn.Embedding(
        #     hps.n_symbols, hps.symbols_embedding_dim)
        # std = sqrt(2.0/(hps.n_symbols+hps.symbols_embedding_dim))
        # val = sqrt(3.0)*std
        # self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()
        #self.cbhg = CBHG()

    def parse_batch(self, batch):
        inputs, input_lengths, mel_targets, targets_lengths, split_infos, embed_targets  = batch

        inputs = mode(inputs).float()
        input_lengths = mode(input_lengths).long()
        mel_targets = mode(mel_targets).float()
        split_infos = mode(split_infos).float()
        targets_lengths = mode(targets_lengths).long()
        #Jared Addition
        embed_targets = mode(embed_targets).float()
    
        # print("input frames shape is {}".format(inputs.shape))
        # print("input_lengths shape is {}".format(input_lengths))
        # print("mel_targets shape is {}".format(mel_targets.shape))
        # print("split_infos shape is {}".format(split_infos.shape))
        # print("split_infos items {}".format(split_infos))
        # print("The target length is {}".format(targets_lengths))
        # print("The embed_targets is {}".format(embed_targets.shape))


        

        return (
            (inputs, input_lengths, mel_targets, embed_targets, targets_lengths),
            (mel_targets, split_infos, targets_lengths))

    def parse_output(self, outputs, output_lengths=None):
        if output_lengths is not None:
            # print(outputs[0]==outputs[1])
            mask = get_mask_from_lengths(output_lengths) # (B, T) Removed True for padding Jared 
            mask = mask.expand(hps.num_mels, mask.size(0), mask.size(1)) # (80, B, T)
            mask = mask.permute(1, 0, 2) # (B, 80, T)
            
            outputs[0].data.masked_fill_(mask, 0.0) # (B, 80, T)
            outputs[1].data.masked_fill_(mask, 0.0) # (B, 80, T)

            
            slice = torch.arange(0, mask.size(2), hps.outputs_per_step)
            # outputs[2].data.masked_fill_(mask[:, 0, slice], 1e3)  # gate energies (B, T//outputs_per_step)
        return outputs

    def forward(self, inputs): #Modified by Jared
        Frame_inputs, input_lengths, mels, embed_targets, output_lengths = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data
        # print("The output lengths is {}".format(output_lengths))
        # print("The input lengths is {}".format(input_lengths))
        # print("shape of mels is {}".format(mels.shape))
        #embedded_inputs = self.embedding(Frame_inputs).transpose(1, 2)
        # Skip embedding Jared
        embedded_inputs = Frame_inputs.float()
        encoder_outputs = self.encoder(embedded_inputs, input_lengths)
        
        ##  append the speaker embedding to encoder Output
        tileable_shape = [-1, 1, hps.speaker_embedding_size]
        # print("embed_targets shape {}".format(embed_targets.shape))
        tileable_embed_targets = embed_targets.reshape(tileable_shape)
        # print("tileable_embed_targets shape {}".format(tileable_embed_targets.shape))
        # print(embed_targets.shape)
        # print(encoder_outputs.shape)
        tiled_embed_targets = tileable_embed_targets.repeat(1, encoder_outputs.shape[1], 1)
        #print("tileable_embed_targets post repeat shape {}".format(tileable_embed_targets.shape))
        encoder_outputs = torch.cat((encoder_outputs, tiled_embed_targets), dim=2)
        # encoder_outputs = encoder_outputs.transpose(1, 2)#Reshape it so it is back to the standard shape of NCHW

        # print("This is the encoder outputs / memory {}".format(encoder_outputs.shape))
        # print("shape of the melspectrograms {}".format(mels.shape))
        ## End of new additions
        # print("This is the encoder outputs {}".format(encoder_outputs.shape))
        mel_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=input_lengths)
        
        # print("The shape of mel_outputs is {}".format(mel_outputs.shape))
        mel_outputs_postnet = self.postnet(mel_outputs)
        
        # print("The shape of mel_outputs_postnet is {}".format(mel_outputs_postnet.shape))
        
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        #post_cbhg = self.cbhg(mel_outputs_postnet)

        #Testing to view named parameters
        # print('The name parameters in the encoder')
        # for name_encoder, _ in self.encoder.named_parameters():
        #     print(name_encoder)
        # print('The name parameters in the decoder')
        # for name_decoder, _ in self.decoder.named_parameters():
        #     print(name_decoder)
        # print('The name parameters in the postnet')
        # for name_postnet, _ in self.postnet.named_parameters():
        #     print(name_postnet)

        list_of_models = [self.encoder, self.decoder, self.postnet]
        list_of_models_name = ["Encoder", "Decoder", "Postnet"]
        trainable_params = []
        trainable_params_names = []
        total_trainable = 0
        total_params = 0
        i=0
        for model in list_of_models:

            trainable_params_filtered = [param for name, param in model.named_parameters() if not (
                                        "bias" in name.lower() or "_projection" in name.lower() or "inputs_embedding" in name.lower()
                                        or "rnn" in name.lower() or "lstm" in name.lower())]
            trainable_params += trainable_params_filtered
        #     trainable_params_filtered_names = [name for name, param in model.named_parameters() if not (
        #                     "bias" in name.lower() or "_projection" in name.lower() or "inputs_embedding" in name.lower()
        #                     or "rnn" in name.lower() or "lstm" in name.lower())]
            
        #     trainable_params_names += trainable_params_filtered_names
        #     pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     pytorch_total_params = sum(p.numel() for p in model.parameters())
        #     print("Number of parameters in {} is {} million".format(list_of_models_name[i], pytorch_total_params_trainable/ 1000000))
        #     total_trainable += pytorch_total_params_trainable
        #     total_params += pytorch_total_params
        #     i+=1

        

        # # Total number of params check
        # print("Total number of parameters is {} million".format(total_params / 1000000))
        # sys.exit("End of test")
        # adding output for output Length
        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, alignments, output_lengths],
            output_lengths), trainable_params

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)

        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, alignments])
        return outputs

    def teacher_infer(self, inputs, mels):
        il, _ =  torch.sort(torch.LongTensor([len(x) for x in inputs]),
                            dim = 0, descending = True)
        text_lengths = mode(il)

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)
        
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, alignments])
