from __future__ import print_function
from keras.callbacks import TensorBoard

from triplet_losses import batch_all_triplet_loss, batch_hard_triplet_loss
from triplet_metrics import triplet_accuracy, mean_norm
from model import build_mnist_model
from dataset.mnist import load_mnist

batch_size = 128
num_classes = 10
epochs = 20

model = build_mnist_model()
print(model.summary())

model.compile(
    loss=batch_all_triplet_loss,
    optimizer='adam',
    metrics=[triplet_accuracy, mean_norm]
)

x_train, y_train, x_test, y_test = load_mnist()

model.fit(x_train, y_train,
          shuffle=True,
          batch_size=batch_size,
          epochs=20,
          verbose=1,
          callbacks=[TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)],
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('./model.h5')
