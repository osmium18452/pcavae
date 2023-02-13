import numpy as np
import matplotlib.pyplot as plt

class DrawTrainMSELoss:
    def __init__(self,init_mse_loss):
        self.mse_loss=np.array([init_mse_loss])

    def add_mse_loss(self,mse_loss):
        self.mse_loss=np.append(self.mse_loss,mse_loss)

    def draw(self,save_file=None):
        fig,ax=plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(5)
        fig.set_dpi(300)
        x=np.arange(self.mse_loss.shape[0])
        ax.plot(x,self.mse_loss,label='MSE Loss')
        fig.legend()
        if save_file is None:
            fig.show()
        else:
            fig.savefig(save_file)
            print('saved to '+save_file)
        plt.close(fig)