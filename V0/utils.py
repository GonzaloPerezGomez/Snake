import matplotlib.pyplot as plt
from IPython import display

def init_plot():
    plt.ion()  # modo interactivo
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    axs[0].set_title("Score")
    axs[1].set_title("Epsilon")
    axs[2].set_title("Loss")
    for ax in axs:
        ax.grid(True)
    plt.tight_layout()
    return fig, axs

def update_plot(fig, axs, scores, epsilons, losses):
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    
    axs[0].plot(scores, label='Score', color='tab:blue')
    axs[1].plot(epsilons, label='Epsilon', color='tab:orange')
    axs[2].plot(losses, label='Loss', color='tab:green')
    
    axs[0].set_title("Score")
    axs[1].set_title("Epsilon")
    axs[2].set_title("Loss")
    
    for ax in axs:
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    display.clear_output(wait=True)
    display.display(fig)
