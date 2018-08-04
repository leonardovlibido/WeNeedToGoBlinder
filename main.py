from train_models import *
import argparse

def main():
    parser = argparse.ArgumentParser()

    help_ = "Use cases"
    parser.add_argument("-uc", "--use_case", help=help_)

    help_ = "Path to dataset"
    parser.add_argument("-dp", "--data_path", help=help_, default='emnist/emnist-balanced-train.csv')

    help_ = "Path to class map file"
    parser.add_argument("-cmpath", "--class_map_path", help=help_, default=None)

    help_ = "Featurizer path"
    parser.add_argument("-featp", "--featurizer_path", help=help_,
                        default='models/autoencoder/autoencoder_64/autoenc_64_encoder_49_0.00.hdf5')

    help_ = "CVAE model name"
    parser.add_argument("-mname", "--model_name", help=help_, default='cvae')

    help_ = "Reconstruction loss (binary_crossentropy or mse)"
    parser.add_argument("-rl", "--reconstruction", help=help_, default='binary_crossentropy')

    help_ = "Latent dimension parameter"
    parser.add_argument("-ldim", "--latent_dim", help=help_, default=64)

    help_ = "Specifies how to choose encoding vector for class (mean, cosine or square)"
    parser.add_argument("-et", "--encoding_type", help=help_, default='mean')

    help_ = "Batch size"
    parser.add_argument("-bs", "--batch_size", help=help_, default=64)

    help_ = "Number of epochs"
    parser.add_argument("-en", "--epochs", help=help_, default=100)

    help_ = "Limit GPU fraction"
    parser.add_argument("-lgpu", "--limit_gpu_fraction", help=help_, default=0.375)

    help_ = "CVAE decoder path"
    parser.add_argument("-cvaedp", "--cvae_decoder_path", help=help_, default=None)

    args = parser.parse_args()
    if args.use_case == 'train_cvae':
        if args.class_map_path:
            cmpath = args.class_map_path
        else:
            cmpath = 'emnist/emnist-balanced-mapping.txt'
        cvae_train(args.data_path, cmpath, args.featurizer_path, args.model_name, args.reconstruction,
                   args.encoding_type,
                   int(args.batch_size), int(args.epochs), float(args.limit_gpu_fraction), int(args.latent_dim))
    elif args.use_case == 'visualize_cvae':
        if args.cvae_decoder_path is None:
            raise ValueError('Must set cvae_decoder_path')
        if args.class_map_path:
            cmpath = args.class_map_path
        else:
            cmpath = None
        cvae_visualize(args.data_path, cmpath, args.featurizer_path, args.cvae_decoder_path, int(args.latent_dim))


if __name__ == "__main__":
    main()
