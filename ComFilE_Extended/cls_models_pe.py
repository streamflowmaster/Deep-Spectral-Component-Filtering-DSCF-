from ComponentFiltering.utils.DSCF_models import SwinBasicEncoder,ResUnetBasicEncoder,PatchMerging,PatchEmbed1D
import torch.nn as nn
import torch
class Hierarchical_1d_cls_model(nn.Module):

    def __init__(self,sig_len = 725, inplanes = 1, outplanes = 2,encoder_name = 'SiT',layers=[2,2,6,2],embed_dim = 32):
        super(Hierarchical_1d_cls_model, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.sig_len = sig_len
        self.encoder_name = encoder_name
        if self.encoder_name == 'SiT':
            self.patch_embed = PatchEmbed1D(patch_size=4,in_chans=inplanes,embed_dim=embed_dim)
            self.inplanes = encoder_name
        self.enc1 = self._make_encoder(block_name=encoder_name, planes=16, blocks=layers[0])
        self.enc2 = self._make_encoder(block_name=encoder_name, planes=32, blocks=layers[1])
        self.enc3 = self._make_encoder(block_name=encoder_name, planes=64, blocks=layers[2])
        self.enc4 = self._make_encoder(block_name=encoder_name, planes=128, blocks=layers[3])
        self.enc = nn.Sequential(
            self.enc1,
            self.enc2,
            self.enc3,
            self.enc4,
        )

        in_features = self._test_feature_size(inplanes)
        self.fc1 = nn.Linear(in_features=in_features,out_features=outplanes)

    def _test_feature_size(self,inplanes):
        test_input = torch.rand((1, inplanes, self.sig_len))
        test_output = self.enc(test_input).reshape(-1)
        return test_output.shape[0]
    def _make_encoder(self, block_name, planes, blocks):

        if block_name == 'SiT':
            layer = SwinBasicEncoder(dim=planes,depth=blocks,num_heads=16,downsample=PatchMerging)

        elif block_name == 'ResUnet':
            layer = ResUnetBasicEncoder(dim=planes,depth=blocks)

        else: layer = None
        if self.inplanes != planes:
            downsample = nn.Conv1d(self.inplanes, planes, stride=1, kernel_size=1, bias=False)
            layers = [downsample, layer]

        else:
            layers = [layer]
        self.inplanes = planes*2

        return nn.Sequential(*layers)

    def forward(self,x):
        B,C,L =x.shape
        # print(L)
        x = self.enc(x).reshape(B,-1)
        return self.fc1(x)


if __name__ == '__main__':
    model = Hierarchical_1d_cls_model(inplanes=1,outplanes=2,encoder_name='SiT',)
    test_input = torch.rand((16, 1, 725))
    output = model(test_input)
    print(output.shape)