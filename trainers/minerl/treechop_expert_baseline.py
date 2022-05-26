import minerl
import numpy as np
from minerl.data import BufferedBatchIter
import tensorflow as tf
from wrappers.minerl.baseline_notebook import dataset_action_batch_to_actions

def train(model):
    data = minerl.data.make('MineRLTreechop-v0')
    iterator = BufferedBatchIter(data, 32000)

    def coiso(batch_size=32, num_epochs=1):
        for current_state, action, reward, next_state, done in iterator.buffered_batch_iter(batch_size, num_epochs):
            x = current_state["pov"].squeeze().astype(np.float32)
            y = dataset_action_batch_to_actions(action)
            yield (x, y)

    optimizer = tf.keras.optimizers.Adam()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer, loss_fn, metrics=[val_acc_metric])

    batch_size = 32
    num_epochs = 50

    model.fit(coiso(batch_size, num_epochs), verbose=1)
    model.save_weights('treechop-fixup')
