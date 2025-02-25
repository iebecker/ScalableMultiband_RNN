
import tensorflow as tf
from ScalableLib.classifier.CustomLayers import SauceLayer, MeanMagLayer
from ScalableLib.mixture.utils import LastRelevantLayer_mod, compute_FinalMeanMags, compute_ColorMatrix

class create_models():
    def __init__(self, multiband_obj):
        # Object from the original multiband implementation.
        # Used for convenience
        self.multiband_obj = multiband_obj
        
    def creat_split_models(self, train_args):
        tf.keras.backend.clear_session()
        inputs = {}
        keys =self.multiband_obj.dataset_test.element_spec[0].keys()
        for key in keys:
            inputs[key] = tf.keras.layers.Input(shape=self.multiband_obj.dataset_test.element_spec[0][key].shape[1:],
                                                dtype=self.multiband_obj.dataset_test.element_spec[0][key].dtype,
                                                name=key
                                                )
        out = {}
        sauces = []
        Mean_Mags = []
        
        for model in range(len(self.multiband_obj.models)):
            # Get the single band outputs
            self.multiband_obj.models[model].trainable = True
            out[model] = self.multiband_obj.models[model].layers[2].output
            slice_model = tf.keras.Model(inputs=self.multiband_obj.models[model].inputs, outputs=out[model])
    
            out_model = slice_model(inputs)
            # Get the last relevant per band per output
            lasts = []
            for i in range(len(out_model)):
                last_relevant = LastRelevantLayer_mod()(out_model[i], inputs['N_'+str(model)])
                lasts.append(last_relevant)
            # Create a Sauce layer per band
            size = len(lasts)
            sauce = SauceLayer(size, name='Sauce_'+str(model))(lasts)
            sauces.append(sauce)
    
            # Compute the mean mags
            MeanMag = MeanMagLayer(self.multiband_obj.w,
                                   name='MeanMag_'+str(model)
                                   )
            mean_mags = MeanMag(inputs['input_LC_'+str(model)],
                                inputs['N_'+str(model)],
                                inputs['M0_'+str(model)],
                                )        
            Mean_Mags.append(mean_mags)
    
        final_MeanMags = [compute_FinalMeanMags(Mean_Mags[i], inputs['N_'+str(i)]) for i in range(self.multiband_obj.n_bands)]
        final_MeanMags = tf.stack(final_MeanMags, axis=1)
        color_matrix = compute_ColorMatrix(self.multiband_obj.n_bands)
        colors = tf.matmul(final_MeanMags, color_matrix)
            
        # Stack the outputs
        embedding = tf.keras.layers.Concatenate(axis=1, name='Concat_Sauces')(sauces)
    
        # Stack the colors
        embedding =  tf.keras.layers.Concatenate(axis=-1, name='Concat_Colors')([embedding, colors])
        # Add dense layers with dropout
        sizes = train_args['fc_layers_central']
        projections = embedding
        for b in range(len(sizes)):
            projections = tf.keras.layers.Dense(sizes[b],
                                                activation=None,
                                                use_bias=False,
                                                )(projections)
            projections = tf.keras.layers.BatchNormalization()(projections, training=True)
            projections = tf.keras.activations.relu(projections)
    
            projections = tf.keras.layers.Dropout(self.multiband_obj.dropout)(projections)
    
        predictions_prob = tf.keras.layers.Dense(self.multiband_obj.num_classes,
                                                 activation='softmax',
                                                 use_bias=True,
                                                 name='Predictions',
                                                 )(projections)
        outputs = {
                'Class': predictions_prob,
    
                }
        model = tf.keras.Model(inputs = inputs, outputs=outputs)
        optimizer = self.multiband_obj.optimizers[0]
        loss = 'categorical_crossentropy'
        model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    
        return model