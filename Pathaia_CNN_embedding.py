from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy
from openslide import OpenSlide
from tqdm import tqdm
import tensorflow as tf
from models import models_CNN


def create_feature_extractor(ModelClass, shape, layer=''):
    model = ModelClass(weights='imagenet', include_top=False, pooling='avg', input_shape=shape)
    if not layer == '':
        layer_model = Model(inputs=model.input,
                            outputs=model.get_layer(layer).output)
        model = Sequential()
        model.add(layer_model)
        model.add(GlobalAveragePooling2D())
    return model


def generator_generator(patches, preproc):
    def generator():
        slide_list = [patches[p]["slide"] for p in patches]
        slide_set = numpy.unique(slide_list)
        slides = {s: OpenSlide(s) for s in slide_set}
        for patch in tqdm(patches):
            slide = slides[patches[patch]["slide"]]
            pil_img = slide.read_region(
                (patches[patch]["x"], patches[patch]["y"]),
                patches[patch]["level"],
                patches[patch]["dimensions"]
            )
            x = numpy.array(pil_img)[:, :, 0:3]
            x = preproc.preprocess_input(x)
            yield x
    return generator


def create_dataset(patches, preproc, patch_size, PREFETCH=None, BATCH=16):
    gen = generator_generator(patches, preproc)
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(numpy.float32),
        output_shapes=((patch_size, patch_size, 3))
    )
    if PREFETCH is not None:
        return dataset.batch(BATCH).prefetch(PREFETCH)
    else:
        return dataset.batch(BATCH)


def get_embeddings(model, patches, patch_size=224, layer=''):
    preproc = models_CNN.models[model]['module']
    ModelClass = models_CNN.models[model]['model']
    model = create_feature_extractor(ModelClass, shape=(patch_size, patch_size, 3), layer=layer)
    patch_set = create_dataset(patches, preproc, patch_size=patch_size)
    descriptors = model.predict(patch_set)
    return descriptors
