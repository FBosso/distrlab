from torch import nn


class Model(nn.Module):
    def __init__(self, component_1, component_2, component_3):
        super().__init__()
        self.comp_1 = component_1
        self.comp_2 = component_2
        self.comp_3 = component_3

    def forward(self, x):
        comp_1_output = self.comp_1(x)
        comp_2_output = self.comp_2(comp_1_output)
        comp_3_output = self.comp_3(comp_2_output)
        return comp_3_output
