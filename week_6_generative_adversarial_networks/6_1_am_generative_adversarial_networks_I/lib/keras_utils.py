from contextlib import contextmanager

@contextmanager
def freeze(model, sub_model):
    for layer in sub_model.layers:
        layer.trainable = False
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    yield # ---->
    for layer in sub_model.layers:
        layer.trainable = True
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
