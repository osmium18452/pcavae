from CVAE import CVAE

class ICVAE:
    def __init__(self, dataloader, latent_size, gpu, learning_rate, gpu_device):
        train_input,train_condition=dataloader.load_cvae_train_data()
        test_input,test_condition=dataloader.load_cvae_test_data()

        print(train_input.shape,train_condition.shape)
        print(test_input.shape,train_condition.shape)

if __name__ == '__main__':
    pass