# Define a CLIP-like dual encoder as the baseline
class DualEncoderWrapper(nn.Module):
    def __init__(self, model):
        super(DualEncoderWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Split the input tensor into `img` and `touch_img`
        img, touch_img = torch.chunk(x, 2, dim=1)
        return self.model(img, touch_img)
