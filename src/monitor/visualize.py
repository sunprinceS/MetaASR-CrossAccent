import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_att_fig(att_map):
    fig = plt.figure(figsize=(16,10))
    plt.imshow(att_map, aspect='auto', origin='upper')
    plt.xlabel('Decoder Index')
    plt.ylabel('Encoder Index')
    plt.colorbar()
    plt.tight_layout()
    return fig
