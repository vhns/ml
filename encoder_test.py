import encoder
import base

train_dataset, test_dataset, validation_dataset = base.dataset_gen('./', 0.1, 0.1, 0.1, 'PUC')

shape = 32
autoencoder = encoder.Autoencoder(8,shape)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

trainmodel(autoencoder, train_dataset, test_dataset, validation_dataset, './tests/decoder/checkpoint.keras', 32, './tests/decoder/')
