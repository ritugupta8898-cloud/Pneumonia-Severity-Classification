import tensorflow as tf

data_dir = "/Users/pratyushgupta/Documents/Pneumonia-Severity-Classification/data/raw/chest_xray/chest_xray/chest_xray"
img_size = (128, 128)
batch_size = 32

# -------- TRAIN DATA (with augmentation) --------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir + "/train",
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary",
    shuffle=True
)
print(train_ds.class_names)

# -------- VALIDATION DATA --------
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir + "/val",
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)

# -------- TEST DATA --------
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir + "/test",
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)

# Normalization Layer
normalization = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization(x), y))
test_ds  = test_ds.map(lambda x, y: (normalization(x), y))

# Prefetch for performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)


