from train_models import *

def main():
    # classifier_train()
    # autoencoder_train()
    # visualize_autoencoder(enc_path='autoenc_checkpoints/autoenc_32_encoder_1_0.11.hdf5',
    #                       dec_path='autoenc_checkpoints/autoenc_32_decoder_1_0.11.hdf5')
    # CVAE_train(model_checkpoint_name='CVAE_autoenc',
    #            model_checkpoint_dir='CVAE_autoenc',
    #            featurizer_path='autoenc_checkpoints/autoenc_32_encoder_33_0.05.hdf5',
    #            limit_gpu_fraction=0.3,
    #            epochs=100)
    visualize_CVAE(feature_gen_path='autoenc_checkpoints/autoenc_32_encoder_33_0.05.hdf5',
                   dec_path='CVAE_autoenc/CVAE_autoenc_decoder_62_0.05.hdf5')
                   # dec_path='CVAE_checkpoints\CVAE_32_decoder_48_0.24.hdf5')
                   # dec_path='CVAE_checkpoints_150\CVAE_32_150_decoder_149_0.24.hdf5')


if __name__ == "__main__":
    main()
