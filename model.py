# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:21:32 2025

@author: Utente
"""
import torch 
import torch.nn.functional as F
import torch.nn as nn

norm_dict = {
    'instance': nn.InstanceNorm3d,
    'batch': nn.BatchNorm3d
}

class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input



  
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, kernel_size=3, 
                 norm_key='instance', dropout_prob=None):
        super(ResidualBlock, self).__init__()
        
        conv = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride,padding= (kernel_size - 1) // 2, bias=True)
                         #padding=(kernel_size - 1) // 2, bias=True)

        norm = norm_dict[norm_key](output_channels, eps=1e-5, affine=True)

        do = Identity() if dropout_prob is None else nn.Dropout3d(p=dropout_prob, inplace=True)

        nonlin = nn.LeakyReLU(inplace=True)

        self.all = nn.Sequential(conv, norm, do, nonlin)

        # downsample residual
        if (input_channels != output_channels) or (stride != 1):
            self.downsample_skip = nn.Sequential(
                nn.Conv3d(input_channels, output_channels, 1, stride, bias=True),
                norm_dict[norm_key](output_channels, eps=1e-5, affine=True), 
            )
        else:
            self.downsample_skip = lambda x: x

    def forward(self, x):
        residual = x

        out = self.all(x)

        residual = self.downsample_skip(x)

        return residual + out


class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=4, dropout_prob=0.2):
        super(SEBlock3D, self).__init__()
        reduced_channels = channels // reduction
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.dropout = nn.Dropout(dropout_prob)  # Aggiungi dropout opzionale
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, depth, height, width = x.size()

        # Squeeze
        squeeze = F.adaptive_avg_pool3d(x, 1).view(batch_size, num_channels)

        # Excitation con riduzione e dropout
        excitation = F.gelu(self.fc1(squeeze))
        excitation = self.dropout(excitation)
        excitation = self.fc2(excitation)

        # Attenzione tramite Sigmoid
        scale = self.sigmoid(excitation).view(batch_size, num_channels, 1, 1, 1)

        return x * scale.expand_as(x)
    
class PositionalEncoding3D(nn.Module):
    def __init__(self, embed_dim, shape):
        super(PositionalEncoding3D, self).__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, embed_dim, *shape))

    def forward(self, x):
        return x + self.positional_embedding.to(x.device)

    
class VolumetricTransformerBlock(nn.Module):
    def __init__(self, in_channels=None, heads=None, dim_head=None, dropout=0.1):
        super(VolumetricTransformerBlock, self).__init__()
        self.in_channels = in_channels
        self.dim_head = dim_head
        self.heads = heads
        self.dropout = dropout

        self.norm1 = None
        self.attn = None
        self.norm2 = None
        self.mlp = None
        self.window_size = [6, 6, 6]
        self.positional_encoding = None  

        self.last_in_channels = None  



        if self.last_in_channels == in_channels:
            return  
        
        self.last_in_channels = in_channels

        if self.heads is None:  # Calcola heads 
            self.dim_head = max(16, in_channels // 8)
            self.heads = in_channels // self.dim_head


        embed_dim = self.heads * self.dim_head
        assert in_channels % self.heads == 0, "in_channels must be divisible by heads"
        
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.heads, dropout=self.dropout)
        self.norm2 = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(self.dropout)
        )
       


    def forward(self, x):
        B, C, H, W, D = x.shape
        
        ws = self.window_size # finestra [6,6,6]
        #Si aggiunge padding a 𝐻, W, D  garantendo che siano multipli di 6
        pad_h = (ws[0] - H % ws[0]) % ws[0]
        pad_w = (ws[1] - W % ws[1]) % ws[1]
        pad_d = (ws[2] - D % ws[2]) % ws[2]
        
        x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))  
        H_p, W_p, D_p = x.shape[2:]  # nuove dim 
        #print(f"Padded shape: {x.shape}")
        
        self.positional_encoding = PositionalEncoding3D(embed_dim=C, shape=(H_p, W_p, D_p)) #viene creata una positional embedding appresa della stessa dimensione dello spazio 3D

        x = self.positional_encoding(x)

        x = x.unfold(2, ws[0], ws[0]).unfold(3, ws[1], ws[1]).unfold(4, ws[2], ws[2])  # unfold per dividere il volume in blocchi tridimensionali: output = (B,C,num_h,num_w,num_d,6,6,6)
        B, C, num_h, num_w, num_d, ws_h, ws_w, ws_d = x.shape
        #print(f"Unfolded shape: {x.shape}")
        #num_h num_w, num_d = numero finestre lungo H,W,D , permute cambia l'ordine delle dimensioni per trattare le finestre come sequenze, reshape combina le finestre in un unico batch
        #con permute ottengo: (B, num_h, num_w, num_d, C, ws_h, ws_w, ws_d) con reshape: (num_windows * B, C, ws_h * ws_w * ws_d) con num_windows = num_h * num_w * num_d
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, C, ws_h * ws_w * ws_d) #--> output: (num_windows×B,C,216) ogni finestra è trattata come una sequenza di 216 token (6x6x6)
        #NB: -1 significa che PyTorch calcolerà la dimensione in modo che il prodotto totale delle dimensioni risulti uguale a quello del tensore originale, eccetto per la dimensione -1
        #print(f"Shape before attention: {x.shape}")
        
        # Attention per window
        x = x.permute(2, 0, 1)  # [seq_len, num_windows * B, C]
        normed_x = self.norm1(x)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x) # self-attention tra voxel della stessa finestra
        #print(f"Attention output shape: {attn_output.shape}")
        
        x = x + attn_output  # Residual connection per facilitare il flusso del gradiente 

        x = x + self.mlp(self.norm2(x)) #Multilayer perceptron non lineare e residual connection 

        
        x = x.permute(1, 2, 0).reshape(B, num_h, num_w, num_d, C, ws_h, ws_w, ws_d)  
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, H_p, W_p, D_p)  # Reshape forma originale 
        

        
        return x[:, :, :H, :W, :D]  # Remove padding

    


# Encoder block in PyTorch
class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = ResidualBlock(in_channels, in_channels * 2)
        self.conv12 = ResidualBlock(in_channels * 2, in_channels * 4)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)


        self.conv2 = ResidualBlock(in_channels * 4, in_channels * 8)
        self.conv21 = ResidualBlock(in_channels * 8, in_channels * 16)

        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=(0, 0, 1))
        self.conv3 = ResidualBlock(in_channels*16, in_channels*32)
        self.conv31 = ResidualBlock(in_channels*32, in_channels*64)

        self.se_block = VolumetricTransformerBlock(in_channels*64)




    def forward(self, x):
        c1 = self.conv1(x)
        #print("shape_after_c1", c1.shape)
        c12 = self.conv12(c1)
        #c12_att = self.se_block12(c12)
        #print("shape_after_c12", c12.shape)
        p1 = self.pool1(c12)
        #print("shape_after_p1", p1.shape)

        c2 = self.conv2(p1)
        #print("shape_after_c2", c2.shape)
        c21 = self.conv21(c2)

        p2 = self.pool2(c21)
        #print("shape_after_p2", p2.shape)
        c3 = self.conv3(p2)
        c31 = self.conv31(c3)

        c31_att = self.se_block(c31)
        return c31_att, c21, c12


    
# Decoder block in PyTorch
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv5 = ResidualBlock(512, 256)
        self.conv51 = ResidualBlock(256, 128)
        self.upconv5 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
       
        self.conv6 = ResidualBlock(128, 64)
        self.conv61 = ResidualBlock(64, 32)
        self.upconv6 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        
        self.conv7 = ResidualBlock(32, 16)


    def forward(self, concatenated, c21_T1CE, c21_T2Flair, c21_T2_T1, c12_T1CE, c12_T2Flair, c12_T2_T1):
        #cs = self.se_block(concatenated) 
        c5 = self.conv5(concatenated) 
        #print("output_c5_shape:", c5.shape)
        c51 = self.conv51(c5)
        #print("output_c51_shape:", c51.shape)
        u5 = self.upconv5(c51)
        #print("output_u5",u5.shape)
        u5c = u5[:, :, :, :, :-1]
        #print("output_u5_cat",u5c.shape)
        u5con = torch.cat([u5c, c21_T1CE, c21_T2Flair, c21_T2_T1], dim=1)
        
        c6 = self.conv6(u5con)
        #print("output_c6",c6.shape)
        c61 = self.conv61(c6)
        #print("output_c61",c61.shape)
        u6 = self.upconv6(c61)
        #print("output_u6",c61.shape)
        u6conc = torch.cat([u6, c12_T1CE, c12_T2Flair, c12_T2_T1], dim=1)

        c7 = self.conv7(u6conc)
        #print("output_c7",c7.shape)
        return c7

# Main V-Net model
class VNetMultiEncoder(nn.Module):
    def __init__(self, input_channels=1):
        super(VNetMultiEncoder, self).__init__()
        self.encoder_T1CE = Encoder(input_channels)
        self.encoder_T2Flair = Encoder(input_channels)
        self.encoder_T2_T1 = Encoder(input_channels * 2)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4 = ResidualBlock(128, 256)
        self.conv41 = ResidualBlock(256, 512)
        self.upconv41 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)

        self.decoder = Decoder()
        self.final_conv = nn.Conv3d(16, 3, kernel_size=1)  # Output 4 classes

    def forward(self, input_T1CE, input_T1, input_T2, input_T2Flair):
        #input_T2_T1 = torch.cat([input_T1, input_T2], dim=1)
        input_T2_T1 = torch.cat([input_T1CE, input_T2Flair], dim=1)

        c31_T1CE, c21_T1CE, c12_T1CE = self.encoder_T1CE(input_T2)
        c31_T2Flair, c21_T2Flair, c12_T2Flair = self.encoder_T2Flair(input_T1)
        c31_T2_T1, c21_T2_T1, c12_T2_T1 = self.encoder_T2_T1(input_T2_T1)
        
  

        p3 = self.pool(c31_T2_T1)
        #print("output_p3", p3.shape)
        c4 = self.conv4(p3)
        #print("output_c4", c4.shape)
        c41 = self.conv41(c4)
        #print("output_c41", c41.shape)

        up41 = self.upconv41(c41)
        #print("output_up41", up41.shape)
        concatenated = torch.cat([up41, c31_T1CE, c31_T2Flair, c31_T2_T1], dim=1)
        #print("output_concatenated:",concatenated.shape)
        c7 = self.decoder(concatenated, c21_T1CE, c21_T2Flair, c21_T2_T1, c12_T1CE, c12_T2Flair, c12_T2_T1)
        outputs = self.final_conv(c7)
        #print("outputs shape:", outputs.shape)
        return outputs