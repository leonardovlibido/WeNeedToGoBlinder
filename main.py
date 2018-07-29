from train_models import *

def main():
    # classifier_train()
    autoencoder_train()
    # visualize_autoencoder(enc_path='autoenc_checkpoints/autoenc_32_encoder_1_0.11.hdf5',
    #                       dec_path='autoenc_checkpoints/autoenc_32_decoder_1_0.11.hdf5')
if __name__ == "__main__":
    main()
