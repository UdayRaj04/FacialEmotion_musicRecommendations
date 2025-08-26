# import os
# import argparse
# import numpy as np
# from glob import glob
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks
# from glob import glob
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# def load_fer_from_dir(data_dir):
#     X, y = [], []
#     for idx, emo in enumerate(EMOTIONS):
#         # pattern = os.path.join(data_dir, "train", emo, "*.png")
#         # files = glob(pattern)
#         pattern = os.path.join(data_dir, "train", emo, "*.*")
#         files = [f for f in glob(pattern) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#         for fp in files:
#             img = tf.keras.preprocessing.image.load_img(fp, color_mode="grayscale", target_size=(48,48))
#             arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#             X.append(arr)
#             y.append(idx)
#     X = np.array(X, dtype=np.float32)
#     y = np.array(y, dtype=np.int64)
#     return X, y

# def build_model(input_shape=(48,48,1), num_classes=7):
#     inputs = layers.Input(shape=input_shape)
#     x = inputs
#     for filters in [32, 64, 128]:
#         x = layers.Conv2D(filters, (3,3), padding="same", activation="relu")(x)
#         x = layers.Conv2D(filters, (3,3), padding="same", activation="relu")(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.MaxPooling2D((2,2))(x)
#         x = layers.Dropout(0.25)(x)

#     x = layers.Flatten()(x)
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.Dropout(0.5)(x)
#     outputs = layers.Dense(num_classes, activation="softmax")(x)

#     model = models.Model(inputs, outputs)
#     model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#     return model

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data-dir", required=True, help="Path to FER dataset (arranged in class folders).")
#     ap.add_argument("--epochs", type=int, default=20)
#     ap.add_argument("--batch-size", type=int, default=64)
#     ap.add_argument("--val-split", type=float, default=0.15)
#     args = ap.parse_args()

#     print("Loading data...")
#     X, y = load_fer_from_dir(args.data_dir)
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_split, stratify=y, random_state=42)

#     print("Building model...")
#     model = build_model()

#     ckpt = callbacks.ModelCheckpoint("model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
#     es = callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

#     print("Training...")
#     # datagen = ImageDataGenerator(
#     # rotation_range=15,
#     # width_shift_range=0.1,
#     # height_shift_range=0.1,
#     # zoom_range=0.1,
#     # horizontal_flip=True
#     # )
#     # history = model.fit(datagen.flow(X_train, y_train, batch_size=args.batch_size),
#     #                 validation_data=(X_val, y_val),
#     #                 epochs=args.epochs,
#     #                 callbacks=[ckpt, es])
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         callbacks=[ckpt, es],
#         verbose=1
#     )

#     print("Saved best model to model.h5")

# if __name__ == "__main__":
#     main()


import os
import argparse
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def load_fer_from_dir(data_dir):
    X, y = [], []
    for idx, emo in enumerate(EMOTIONS):
        pattern = os.path.join(data_dir, "train", emo, "*.*")
        files = [f for f in glob(pattern) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for fp in files:
            img = tf.keras.preprocessing.image.load_img(fp, color_mode="grayscale", target_size=(48,48))
            arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            X.append(arr)
            y.append(idx)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

def build_model(input_shape=(48,48,1), num_classes=7):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Deeper CNN with more filters
    for filters in [64, 128, 256]:
        x = layers.Conv2D(filters, (3,3), padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, (3,3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Path to FER dataset (arranged in class folders).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-split", type=float, default=0.15)
    args = ap.parse_args()

    print("Loading data...")
    X, y = load_fer_from_dir(args.data_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_split, stratify=y, random_state=42)

    print("Building model...")
    model = build_model()

    ckpt = callbacks.ModelCheckpoint("model_best.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
    es = callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True)
    lr_sched = callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1)

    print("Training with data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        callbacks=[ckpt, es, lr_sched],
        verbose=1
    )

    print("Saved best model to model_best.keras")

if __name__ == "__main__":
    main()




# # {
#   "angry": {
#     "youtube": [
#       {
#         "title": "Relaxing Piano for Anger Relief",
#         "youtubeId": "1ZYbU82GVz4"
#       },
#       {
#         "title": "Chillhop - Calm Beats",
#         "youtubeId": "5yx6BWlEVcY"
#       }
#     ],
#     "spotify": [
#       {
#         "title": "Peaceful Piano",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DX4sWSpwq3LiO"
#       },
#       {
#         "title": "Lo-Fi Beats",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DXc8kgYqqlMhw"
#       }
#     ]
#   },
#   "disgust": {
#     "youtube": [
#       {
#         "title": "Ambient Study Music",
#         "youtubeId": "wpqT7vNDMfc"
#       },
#       {
#         "title": "Deep Focus",
#         "youtubeId": "FtmQju-Pq9g"
#       }
#     ],
#     "spotify": [
#       {
#         "title": "Deep Focus",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DWZeKCadgRdKQ"
#       },
#       {
#         "title": "Focus Flow",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DXc2aPBXGmXrt"
#       }
#     ]
#   },
#   "fear": {
#     "youtube": [
#       {
#         "title": "Comfort & Uplift",
#         "youtubeId": "lFcSrYw-ARY"
#       },
#       {
#         "title": "Calm Your Mind",
#         "youtubeId": "DWcJFNfaw9c"
#       }
#     ],
#     "spotify": [
#       {
#         "title": "Songs to Sing in the Shower",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DWSqmBTGDYngZ"
#       },
#       {
#         "title": "Confidence Boost",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DX4fpCWaHOned"
#       }
#     ]
#   },
#   "happy": {
#     "youtube": [
#       {
#         "title": "Good Vibes Only",
#         "youtubeId": "f02mOEt11OQ"
#       },
#       {
#         "title": "Feel Good Indie",
#         "youtubeId": "gWcF6P9Vxzs"
#       }
#     ],
#     "spotify": [
#       {
#         "title": "Feel Good Friday",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DXdPec7aLTmlC"
#       },
#       {
#         "title": "Happy Hits!",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DXdPec7aLTmlC"
#       }
#     ]
#   },
#   "sad": {
#     "youtube": [
#       {
#         "title": "Cheer Up Mix",
#         "youtubeId": "p2H7r8r7Ruo"
#       },
#       {
#         "title": "Upbeat Pop",
#         "youtubeId": "9m0Wb9G9Zxw"
#       }
#     ],
#     "spotify": [
#       {
#         "title": "Life Sucks",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DX7gIoKXt0gmx"
#       },
#       {
#         "title": "Feelin' Good",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DX1g0iEXLFycr"
#       }
#     ]
#   },
#   "surprise": {
#     "youtube": [
#       {
#         "title": "Discover Weekly Vibes",
#         "youtubeId": "tNkZsRW7h2c"
#       },
#       {
#         "title": "Eclectic Mix",
#         "youtubeId": "c7rCyll5AeY"
#       }
#     ],
#     "spotify": [
#       {
#         "title": "Pollen",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DWWBHeXOYZf74"
#       },
#       {
#         "title": "All New Music Friday",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DX4JAvHpjipBk"
#       }
#     ]
#   },
#   "neutral": {
#     "youtube": [
#       {
#         "title": "Lo-Fi Beats",
#         "youtubeId": "jfKfPfyJRdk"
#       },
#       {
#         "title": "Coding Music",
#         "youtubeId": "2YllipGl0aU"
#       }
#     ],
#     "spotify": [
#       {
#         "title": "Lo-Fi Beats",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DXd9rSDyQguIk"
#       },
#       {
#         "title": "Chill Vibes",
#         "spotifyUri": "spotify:playlist:37i9dQZF1DX889U0CL85jj"
#       }
#     ]
#   }
# }