from train_models import *

def main():
    # classifier_train()
    autoencoder_train()
    # visualize_autoencoder(enc_path='autoenc_checkpoints/autoenc_32_encoder_1_0.11.hdf5',
    #                       dec_path='autoenc_checkpoints/autoenc_32_decoder_1_0.11.hdf5')
    # CVAE_train()
    # visualize_CVAE(feature_gen_path='best_model_classifier/aug_32.43-0.28.hdf5',
    #                dec_path='CVAE_checkpoints_crossentropy/CVAE_32_decoder_49_0.29.hdf5')
                   # dec_path='CVAE_checkpoints\CVAE_32_decoder_48_0.24.hdf5')
                   # dec_path='CVAE_checkpoints_150\CVAE_32_150_decoder_149_0.24.hdf5')


if __name__ == "__main__":
    main()
