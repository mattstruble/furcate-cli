import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import furcate


from load_data import load_data
load_data()

def build_inceptionV3():
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    local_weights_file = 'data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pretrained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
    pretrained_model.load_weights(local_weights_file)

    for layer in pretrained_model.layers:
        layer.trainable = False

    last_output = pretrained_model.get_layer('mixed7').output

    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(pretrained_model.input, x)

    return model


def build_cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


class App(furcate.Fork):

    def get_model(self):
        if self.model_name == 'InceptionV3':
            return build_inceptionV3()
        else:
            return build_cnn()

    def get_metrics(self):
        return ['accuracy']

    def get_optimizer(self):
        return tf.keras.optimizers.RMSprop(lr=self.learning_rate)

    def get_loss(self):
        return 'binary_crossentropy'

    def get_filepaths(self):
        return 'data/training', None, 'data/validation'

    def get_datasets(self, train_fp, test_fp, valid_fp):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rotation_range=40,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_fp,
            batch_size=self.batch_size,
            target_size=(150, 150),
            class_mode='binary'
        )

        validation_generator = test_datagen.flow_from_directory(
            valid_fp,
            batch_size=self.batch_size,
            target_size=(150, 150),
            class_mode='binary'
        )

        return train_generator, None, validation_generator

if __name__=="__main__":
    app = App('config.json')
    app.run()