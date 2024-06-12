import os
import json
import os
import pickle
from typing import Union, List, Any

import ScalableLib.base.Multiband as Multiband
import ScalableLib.base.Parser as Parser
import ScalableLib.base.plot as plot
import numpy as np
import tensorflow as tf

from ScalableLib.classifier.CustomLayers import *
from ScalableLib.classifier.CustomLosses import *
from ScalableLib.classifier.CustomMetrics import *
from ScalableLib.classifier.CustomModels import CustomModelBand, CustomModelCentral
from pandas import DataFrame
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

from ScalableLib.classifier.CustomLayers import MeanMagLayer, RawTimesLayer, RNNLayersBands, SauceLayer, ApplyMask, \
    InputCentral, MeanColorLayer, AllTimes, RNNLayersCentral, LastRelevantLayer
from ScalableLib.classifier.CustomLosses import CrossEntropy_FullWeights, MSE_masked
from ScalableLib.classifier.CustomMetrics import CustomAccuracy, CustomTopKAccuracy, CustomFinalAccuracy, \
    CustomTopKFinalAccuracy, CustomFinalF1Score, Masked_RMSE, Masked_R2


class Network(Multiband.Network):
    def __init__(self,
                 **kwargs):
        super(Network).__init__(**kwargs)
        self.regression_scores = None
        self.test_results = None
        self.output_params = None
        self.scalers = None
        self.step = None

    def __add_model_band(self, i):
        # Extract the streaming mean magnitudes
        mean_mag = MeanMagLayer(self.w,
                                name='MeanMag_' + str(i)
                                )
        mean_mags = mean_mag(self.inputs['input_LC_' + str(i)],
                             self.inputs['N_' + str(i)],
                             self.inputs['M0_' + str(i)],
                             )
        # Extract the raw times
        RawTimes = RawTimesLayer(self.w,
                                 name='RawTimes_' + str(i),
                                 )
        raw_times = RawTimes(self.inputs['input_LC_' + str(i)],
                             self.inputs['N_' + str(i)],
                             self.inputs['T0_' + str(i)],
                             )

        # Define and run the RNN. Extract the output
        self.RNNs[i] = RNNLayersBands(hidden_sizes=self.size_hidden_bands,
                                      index=i,
                                      common_kernel_layer=self.common_kernel,
                                      common_recurrent_kernel_layer=self.common_recurrent_kernel,
                                      bidirectional=self.bidirectional_band,
                                      use_mod_cell=self.use_common_layers,
                                      use_gated_common=self.use_gated_common,
                                      )

        self.rnn_outputs[i] = self.RNNs[i](self.inputs['input_LC_' + str(i)],
                                           self.inputs['N_' + str(i)]
                                           )
        sauce_layer = SauceLayer(len(self.rnn_outputs[i]), name='Scale_Layer_band_' + str(i))
        self.sauce[i] = sauce_layer(self.rnn_outputs[i])

        # Get project to num_Classes dimensions
        projections = self.sauce[i]
        for j in range(len(self.fc_layers_bands)):
            # Traditional BatchNorm
            projections = tf.keras.layers.Dense(self.fc_layers_bands[j],
                                                activation=None,
                                                use_bias=False,
                                                )(projections)
            projections = tf.keras.layers.BatchNormalization()(projections, training=True)
            projections = tf.keras.activations.relu(projections)

            projections = tf.keras.layers.Dropout(self.dropout)(projections)
        # Add softmax layer
        self.predictions_prob[i] = tf.keras.layers.Dense(self.num_classes,
                                                         activation='softmax',
                                                         use_bias=True,
                                                         name='Predictions',
                                                         bias_initializer=tf.constant_initializer(
                                                             1.0 / self.numpy_weights),
                                                         )(projections)

        ApplyMasks = ApplyMask(self.num_classes,
                               mask_value=self.mask_value,
                               name='Class')
        masked_prediction = ApplyMasks(self.predictions_prob[i], self.inputs['N_' + str(i)])

        self.outputs_[i] = {
            'Class': masked_prediction,
            'Sauce_' + str(i): self.rnn_outputs[i],
            'MeanMag_' + str(i): mean_mags,
            'RawTimes_' + str(i): raw_times,
        }

        # Define the input and output signature
        input_sig_0 = {'ID': tf.TensorSpec(shape=(None,), dtype=tf.string)}
        for bb in range(self.n_bands):
            input_sig_0['input_LC_' + str(bb)] = tf.TensorSpec(shape=(None, None, self.w), dtype=tf.float32)
            input_sig_0['O_' + str(bb)] = tf.TensorSpec(shape=(None, None), dtype=tf.int32)
            input_sig_0['N_' + str(bb)] = tf.TensorSpec(shape=(None,), dtype=tf.int32)
            input_sig_0['M0_' + str(bb)] = tf.TensorSpec(shape=(None,), dtype=tf.float32)
            input_sig_0['T0_' + str(bb)] = tf.TensorSpec(shape=(None,), dtype=tf.float32)
            input_sig_0['U_' + str(bb)] = tf.TensorSpec(shape=(None, None,), dtype=tf.float32)

        input_sig_1 = {'Class': tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.int32)}

        self.models[i] = CustomModelBand(inputs=self.inputs,
                                         outputs=self.outputs_[i],
                                         name='Model_' + str(i),
                                         signature=(input_sig_0, input_sig_1),
                                         N_skip=self.N_skip,
                                         )

        self.loss_functions[i] = {
            'Class': CrossEntropy_FullWeights(N_skip=self.N_skip,
                                              mask_value=self.mask_value,
                                              ),
        }

        self.train_metrics[i] = {'Class': [CustomAccuracy(name='Acc', N_skip=self.N_skip, mask_value=self.mask_value),
                                           CustomTopKAccuracy(k=2, name='Top2', N_skip=self.N_skip,
                                                              mask_value=self.mask_value), ]
                                 }

        self.optimizers[i] = self.__get_optim(self.lr_bands[i], optimizer='AdamW')

        self.models[i].compile(loss=self.loss_functions[i],
                               optimizer=self.optimizers[i],
                               metrics=self.train_metrics[i],
                               )

    def __add_model_central(self):
        outputs_bands: Union[List[None], Any] = [None] * self.n_bands
        sauce_layers: Union[List[None], Any] = [None] * self.n_bands
        sauce = [None] * self.n_bands
        orders = [None] * self.n_bands
        outs = [None] * self.n_bands
        mean_mags = [None] * self.n_bands
        raw_times = [None] * self.n_bands

        for j in range(self.n_bands):
            self.models[j].trainable = False

            # Get output of the models, to use as input for the spine network.
            outs[j] = self.models[j](self.inputs_central)

            # Get output projections
            orders[j] = self.inputs_central['O_' + str(j)]

            # Get the sauce
            outputs_bands[j] = outs[j]['Sauce_' + str(j)]

            # Get the MeanMag per timestep per band
            mean_mags[j] = outs[j]['MeanMag_' + str(j)]

            # Get the raw time per timestep per band
            raw_times[j] = outs[j]['RawTimes_' + str(j)]

            # Concatenate the raw input to each band
            if self.use_raw_input_central:
                sauce_layers[j] = SauceLayer(len(outputs_bands[j]) + 1, name='SauceLayer_Central_' + str(j))
                projection_raw_input = tf.keras.layers.Dense(self.size_hidden_bands[0])
                projected_raw_input = projection_raw_input(self.inputs_central['input_LC_' + str(j)])

                sauce[j] = sauce_layers[j]([projected_raw_input] + outputs_bands[j])
            else:  # Do not use raw input
                # Give control of the weighting to the spine
                sauce_layers[j] = SauceLayer(len(outputs_bands[j]), name='SauceLayer_Central_' + str(j))
                sauce[j] = sauce_layers[j](outputs_bands[j])

            # Perform LayerNorm on the inputs
            sauce[j] = tf.keras.layers.LayerNormalization(name='LayerNorm_Sauce_' + str(j))(sauce[j])

            # Compute the translation layer if applicable
            if self.use_output_layers_bands:
                # sauce[j] = self.band_output_kernels[j](sauce[j])
                sauce[j] = tf.keras.layers.Dense(self.size_hidden_bands[-1],
                                                 activation='relu',
                                                 use_bias=True,
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='glorot_uniform',
                                                 name='Band_Output_Layer_1_' + str(j)
                                                 )(sauce[j])
                # sauce[j] = tf.keras.layers.Dropout(self.dropout)(sauce[j])
                # sauce[j] = tf.keras.layers.LayerNormalization()(sauce[j])
                sauce[j] = tf.keras.layers.Dense(self.size_hidden_bands[-1],
                                                 activation='relu',
                                                 use_bias=True,
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='glorot_uniform',
                                                 name='Band_Output_Layer_2_' + str(j)
                                                 )(sauce[j])
                sauce[j] = tf.keras.layers.Dense(self.size_hidden_bands[-1],
                                                 activation='relu',
                                                 use_bias=True,
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='glorot_uniform',
                                                 name='Band_Output_Layer_3_' + str(j)
                                                 )(sauce[j])

        # Concat the sauces, along the axis 1.
        ns = [self.inputs_central['N_' + str(i)] for i in range(self.n_bands)]
        input_central_layer = InputCentral(name='Input_Central')
        sorted_states, n_central = input_central_layer(sauce, orders, ns)

        # Compute the colors per time-step
        mean_colors = MeanColorLayer(self.n_bands, name='Mean_Color')
        colors = mean_colors(mean_mags, ns, orders)

        # Compute the deltas over the entire time
        compute_deltas = AllTimes(self.n_bands, name='AllTimes_Central')
        delta_times = compute_deltas(raw_times, n_central)

        # Perform LayerNorm on the sorted states
        sorted_states = tf.keras.layers.LayerNormalization()(sorted_states)

        # Make the spine
        rnn_central = RNNLayersCentral(self.size_hidden_central,
                                       bidirectional=self.bidirectional_central,
                                       name='RNN_Central'
                                       )

        output_central = rnn_central(sorted_states,
                                     n_central,
                                     )

        # Compute the sauce for the central RNN
        sauce_central = SauceLayer(len(output_central), name='Final_Sauce')
        output_central = sauce_central(output_central)

        # Concatenate the mean colors and times to the sauce output
        output_central = tf.keras.layers.Concatenate(axis=-1, name='Concat_info')([output_central,
                                                                                   colors,
                                                                                   delta_times])

        dense_central = output_central
        for k in range(len(self.fc_layers_central)):
            dense_central = tf.keras.layers.Dense(self.fc_layers_central[k],
                                                  activation=None,
                                                  use_bias=False,
                                                  )(dense_central)
            dense_central = tf.keras.layers.BatchNormalization(name='Final_BN_' + str(k))(dense_central, training=True)
            dense_central = tf.keras.layers.ReLU(name='Final_Dense_' + str(k))(dense_central)

        # Compute the predictions of the classifier
        d_predictions_central = tf.keras.layers.Dense(self.num_classes,
                                                      activation='softmax',
                                                      use_bias=True,
                                                      name='Prediction',
                                                      bias_initializer=tf.constant_initializer(1.0 / self.numpy_weights)
                                                      )
        prediction_central = d_predictions_central(dense_central)

        # Everything masked is self.mask_value
        apply_masks = ApplyMask(self.num_classes,
                                mask_value=self.mask_value,
                                name='Class')
        masked_prediction_central = apply_masks(prediction_central, n_central)

        # Get last prediction
        last_relevant = LastRelevantLayer(name='FinalClass')
        last_prediction_central = last_relevant(prediction_central, n_central)

        # Get last relevant output
        # TODO: Clean the last_relevant function appearances
        last_output_central = self.__last_relevant(output_central, n_central)

        # Compute the physical parameter estimation
        last_phys = {}
        for param in self.physical_params:

            output = last_output_central
            for l in range(len(self.regression_size)):
                # Add the layer
                output = tf.keras.layers.Dense(self.regression_size[l],
                                               activation=None,
                                               use_bias=False,
                                               name='Dense_' + str(l) + '_' + param
                                               )(output)
                output = tf.keras.layers.BatchNormalization(name='BN_' + str(l) + '_' + param
                                                            )(output, training=True)
                output = tf.keras.activations.relu(output)

            last_phys[param] = tf.keras.layers.Dense(1,
                                                     activation=None,
                                                     use_bias=True,
                                                     name='Pred_' + param)(output)

        self.outputs_end = {
            'Class': masked_prediction_central,
            'FinalClass': last_prediction_central,
        }
        # Set as output the physical parameter estimation
        for param in self.physical_params:
            self.outputs_end[param] = last_phys[param]

        # Define the input and output signature

        input_sig_0 = {'ID': tf.TensorSpec(shape=(None,), dtype=tf.string)}
        for bb in range(self.n_bands):
            input_sig_0['input_LC_' + str(bb)] = tf.TensorSpec(shape=(None, None, self.w), dtype=tf.float32)
            input_sig_0['O_' + str(bb)] = tf.TensorSpec(shape=(None, None), dtype=tf.int32)
            input_sig_0['N_' + str(bb)] = tf.TensorSpec(shape=(None,), dtype=tf.int32)
            input_sig_0['M0_' + str(bb)] = tf.TensorSpec(shape=(None,), dtype=tf.float32)
            input_sig_0['T0_' + str(bb)] = tf.TensorSpec(shape=(None,), dtype=tf.float32)
            input_sig_0['U_' + str(bb)] = tf.TensorSpec(shape=(None, None,), dtype=tf.float32)

        input_sig_1 = {'Class': tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.int32),
                       'FinalClass': tf.TensorSpec(shape=(None, self.num_classes), dtype=tf.int32)
                       }
        for param in self.physical_params:
            input_sig_1[param] = tf.TensorSpec(shape=(None,), dtype=tf.float32)

        self.model_central = CustomModelCentral(inputs=self.inputs_central,
                                                outputs=self.outputs_end,
                                                signature=(input_sig_0, input_sig_1),
                                                n_bands=self.n_bands,
                                                N_skip=self.N_skip,
                                                name='Model_central',
                                                )

        self.optimizers['Central'] = self.__get_optim(self.lr_central, optimizer='AdamW')

        self.loss_functions['Central'] = {
            'Class': CrossEntropy_FullWeights(N_skip=self.N_skip,
                                              mask_value=self.mask_value,
                                              ),
            'FinalClass': None,
        }
        for param in self.physical_params:
            self.loss_functions['Central'][param] = MSE_masked(mask_value=self.mask_value)

        self.train_metrics['Central'] = {
            'Class': [CustomAccuracy(name='CentralAcc', N_skip=self.N_skip, mask_value=self.mask_value),
                      CustomTopKAccuracy(k=2, name='CentralTop2', N_skip=self.N_skip, mask_value=self.mask_value),
                      CustomFinalAccuracy(name='FinalAcc', mask_value=self.mask_value),
                      CustomTopKFinalAccuracy(k=2, name='FinalTop2', mask_value=self.mask_value),
                      CustomFinalF1Score(self.num_classes, name='Final_FScore', mask_value=self.mask_value)
                      ],

        }
        for param in self.physical_params:
            self.train_metrics['Central'][param] = [Masked_RMSE(name='Masked_RMSE', mask_value=self.mask_value),
                                                    Masked_R2(name='Masked_R2', mask_value=self.mask_value),
                                                    ]

        self.model_central.compile(loss=self.loss_functions['Central'],
                                   loss_weights=self.loss_weights_central,
                                   optimizer=self.optimizers['Central'],
                                   metrics=self.train_metrics['Central']
                                   )

    def __add_models(self):
        # Create common matrices
        if self.use_common_layers:
            self.__create_common_layers()
        # Create transformations of each RNN
        if self.use_output_layers_bands:
            self.__create_output_layers()

        # Add band models
        for i in range(self.n_bands):
            self.__add_model_band(i)
        # Add spine model
        self.__add_model_central()

    def train(self, train_args, tfrecords_train, tfrecords_val, tfrecords_test):
        self.set_train_settings(train_args)
        # Load scalers
        if 'regression' in self.mode:
            # Load the scalers
            with open(self.path_scalers, 'rb') as file:
                self.scalers = pickle.load(file)

        self.__initialize_datasets(tfrecords_train, tfrecords_val, tfrecords_test)
        self.__define_inputs()
        self.__add_placeholders()
        self.__add_writers()
        self.__add_models()
        self.__add_callbacks()

    def train_logic(self, input, target):
        """This train step controls the logic of the training of all the models."""
        # Train the scapulars

        # Boolean to finish training once everything early topped
        self.boolean_EarlyStopping = True

        # Create an input without physical parameters for the band models
        target_band = {'Class': target['Class']}
        for b in range(self.n_bands):
            # Update boolean_EarlyStopping
            self.boolean_EarlyStopping = self.boolean_EarlyStopping and self.models[b].stop_training
            # If not early stopped
            if not self.models[b].stop_training:
                self.models[b].trainable = True
                self.models[b].reset_states()  # Reset states after each batch

                self.logs_train[b] = self.models[b].train_step(input, target_band)
                self.models[b].trainable = False

        # Train the spine
        # Update boolean_EarlyStopping
        self.boolean_EarlyStopping = self.boolean_EarlyStopping and self.model_central.stop_training

        # If not early stopped
        if not self.boolean_EarlyStopping:
            # Start training after step_wait steps
            if self.step > self.steps_wait or self.step == 0:
                target['FinalClass'] = tf.constant([[0] * self.num_classes], dtype=tf.int32)
                # Train multiple times per step
                for _ in range(self.train_steps_central):
                    self.model_central.reset_states()
                    self.log_train_central = self.model_central.train_step(input, target)
                if self.model_central.stop_training:
                    self.boolean_EarlyStopping = True

    def train_loop(self):
        """Function to perform the training process"""
        # Run callbacks on train begin
        self.callbacks['Central'].on_train_begin()
        for b in range(self.n_bands):
            self.callbacks[b].on_train_begin()
        print('Start training')
        try:
            batch = 0
            # Start the main loop
            for epoch in range(self.epochs):
                self.epoch = epoch
                for input_, target in self.dataset_train:
                    self.step = batch
                    # Run the train step
                    self.train_logic(input_, target)
                    batch += 1

                # Run validation on epoch end
                # Store train and validation metrics
                self.val_loop()
                # Run callbacks on epoch end
                self.callbacks['Central'].on_epoch_end(epoch=epoch,
                                                       logs=self.log_val_central,
                                                       )

                for b in range(self.n_bands):
                    self.callbacks[b].on_epoch_end(epoch=epoch,
                                                   logs=self.logs_val[b],
                                                   )
                if self.boolean_EarlyStopping:
                    print('Early Stopping')
                    break

        except KeyboardInterrupt:
            pass
        finally:
            # Run callbacks on train end
            self.callbacks['Central'].on_train_end()
            for b in range(self.n_bands):
                self.callbacks[b].on_train_end()

            # Save preprocessing and train data
            self.save_setup()
            # Save weights
            self.save_weights()

            # Run test
            self.test_loop(print_report=self.print_report)

    def val_loop(self):

        # Write logs
        for j in range(self.n_bands):
            # Write train logs
            with self.train_summary_writers[j].as_default():
                tf.summary.scalar('Classifier_Loss_train', self.logs_train[j]['train_loss'], step=self.epoch)
                tf.summary.scalar('Classifier_Accuracy_train', self.logs_train[j]['Class_Acc'], step=self.epoch)

                tf.summary.scalar('Stop Training', self.models[j].stop_training, step=self.epoch)
            # Evaluate bands val
            self.models[j].reset_states()  # Reset states after each batch
            self.logs_val[j] = self.models[j].evaluate(self.dataset_val,
                                                       return_dict=True,
                                                       )
            with self.val_summary_writers[j].as_default():
                tf.summary.scalar('Classifier_Loss_val', self.logs_val[j]['Class_loss'], step=self.epoch)
                tf.summary.scalar('Classifier_Accuracy_val', self.logs_val[j]['Class_Acc'], step=self.epoch)

        with self.train_summary_writer_C.as_default():
            tf.summary.scalar('Classifier_Loss_train', self.log_train_central['train_loss'], step=self.epoch)
            tf.summary.scalar('Classifier_Accuracy_train', self.log_train_central['Class_CentralAcc'], step=self.epoch)
            tf.summary.scalar('Classifier_Top2Accuracy', self.log_train_central['Class_CentralTop2'], step=self.epoch)
            tf.summary.scalar('Classifier_FinalAccuracy', self.log_train_central['Class_FinalAcc'], step=self.epoch)
            tf.summary.scalar('Classifier_FinalTop2Accuracy', self.log_train_central['Class_FinalTop2'],
                              step=self.epoch)
            tf.summary.scalar('Classifier_Final_FScore', self.log_train_central['Class_Final_FScore'], step=self.epoch)

            tf.summary.scalar('Stop Training', self.model_central.stop_training, step=self.epoch)
            for param in self.physical_params:
                tf.summary.scalar('Regression_Masked_RMSE_' + param,
                                  self.log_train_central['Pred_' + param + '_' + 'Masked_RMSE'], step=self.epoch)
                tf.summary.scalar('Regression_Masked_R2_' + param,
                                  self.log_train_central['Pred_' + param + '_' + 'Masked_R2'], step=self.epoch)

        with self.val_summary_writer_C.as_default():
            self.model_central.reset_states()  # Reset states after each batch
            self.log_val_central = self.model_central.evaluate(self.dataset_val,
                                                               return_dict=True,
                                                               )
            tf.summary.scalar('Classifier_Loss_val', self.log_val_central['loss'], step=self.epoch)
            tf.summary.scalar('Classifier_Accuracy_val', self.log_val_central['Class_CentralAcc'], step=self.epoch)
            tf.summary.scalar('Classifier_Top2Accuracy', self.log_val_central['Class_CentralTop2'], step=self.epoch)
            tf.summary.scalar('Classifier_FinalAccuracy', self.log_val_central['Class_FinalAcc'], step=self.epoch)
            tf.summary.scalar('Classifier_FinalTop2Accuracy', self.log_val_central['Class_FinalTop2'], step=self.epoch)
            tf.summary.scalar('Classifier_Final_FScore', self.log_val_central['Class_Final_FScore'], step=self.epoch)
            for param in self.physical_params:
                tf.summary.scalar('Regression_Masked_RMSE_' + param,
                                  self.log_val_central['Pred_' + param + '_' + 'Masked_RMSE'], step=self.epoch)
                tf.summary.scalar('Regression_Masked_R2_' + param,
                                  self.log_val_central['Pred_' + param + '_' + 'Masked_R2'], step=self.epoch)

    def test_loop(self, print_report=True):
        """Run the test loop."""
        # Ground truth and ID
        id_ = []
        Class = []
        prob = []
        # Physical parameter containers
        phys_params = {}
        if 'regression' in self.mode:
            for param in self.physical_params:
                phys_params['Pred_' + param] = []
                phys_params[param] = []

        for batch in self.dataset_test:
            # Compute the prediction
            pred = self.model_central(batch[0])

            # Get ID and Class from the input
            id_.append(batch[0]['ID'].numpy())
            Class.append(batch[1]['Class'].numpy().argmax(axis=1))
            # Get the final prediction probability
            prob.append(pred['FinalClass'])

            # Get the regression results, if applicable
            if 'regression' in self.mode:
                for param in self.physical_params:
                    phys_params['Pred_' + param].append(pred[param])
                    phys_params[param].append((batch[1][param]))

        # Flatten
        id_ = np.array([j for i in id_ for j in i]).astype(str)
        Class = np.array([self.trans[j] for i in Class for j in i])
        prob = np.array([j for i in prob for j in i])

        if 'regression' in self.mode:
            for param in self.physical_params:
                phys_params['Pred_' + param] = np.array([j for i in phys_params['Pred_' + param] for j in i])
                phys_params[param] = np.array([j for i in phys_params[param] for j in i])

        # Predict
        output = {'Probability': np.array(prob), 'ID': id_, 'Class': Class}

        # Add everything into the output dict_transform
        output['Prediction'] = np.vectorize(self.trans.get)(output['Probability'].argmax(axis=1))
        # Transform the phys params to un normalized values
        mask_value = {}  # dict_transform to store the values representing the mask.
        if 'regression' in self.mode:
            # Load the scalers
            with open(self.path_scalers, 'rb') as file:
                self.scalers = pickle.load(file)
            # Transform each column
            self.output_params = {}
            for param in self.physical_params:
                # Value of the mask is always zero
                mask_value[param] = self.mask_value
                # Invert the predictions and the ground truth
                output['Pred_' + param] = self.scalers[param].inverse_transform(
                    phys_params['Pred_' + param].reshape(-1, 1))
                output[param] = self.scalers[param].inverse_transform(phys_params[param].reshape(-1, 1))

                # Transform the masked value
                mask_value[param] = self.scalers[param].inverse_transform(mask_value[param])

                # Store the output in a dict_transform
                self.output_params['Pred_' + param] = output['Pred_' + param].ravel()
                self.output_params[param] = output[param].ravel()

        self.test_results = output

        if print_report:
            print(classification_report(self.test_results['Class'], self.test_results['Prediction']))

            if 'regression' in self.mode:
                self.regression_scores = {'R2': {}, 'RMSE': {}}
                for param in self.physical_params:
                    # Not all the values have measured parameters.
                    # Estimate the metrics using the existint phys params
                    mask = output[param] > mask_value[param] + 1
                    self.regression_scores['R2'][param] = r2_score(output[param][mask], output['Pred_' + param][mask])
                    self.regression_scores['RMSE'][param] = mean_squared_error(output[param][mask],
                                                                               output['Pred_' + param][mask],
                                                                               squared=False)  # False means RMSE

                print(self.regression_scores)

    def save_results(self, df_paths):
        """Save the results to the selected path. If none, do nothing"""
        if df_paths is not None:
            res = DataFrame(self.test_results['Probability'], index=self.test_results['ID'])
            res.columns = 'Prob_' + res.columns.map(self.trans)
            res = res.assign(**{'Class': self.test_results['Class'], 'Pred': self.test_results['Prediction']})

            if 'regression' in self.mode:
                for param in self.physical_params:
                    true_phys = 'True_' + param
                    true_col = self.output_params[param]

                    pred_phys = 'Pred_' + param
                    pred_col = self.output_params[pred_phys]

                    res = res.assign(**{true_phys: true_col, pred_phys: pred_col})
            res.to_csv(df_paths, index=True, index_label=False)

    def __add_callbacks(self):

        # Initialize callbacks for all the models
        self.callbacks = {}
        # For spine
        early_stopping_c = EarlyStopping(monitor='Class_FinalAcc',
                                         **self.callbacks_args
                                         )
        self.callbacks['Central'] = tf.keras.callbacks.CallbackList([early_stopping_c],
                                                                    model=self.model_central,
                                                                    )

        # For each scapular
        for b in range(self.n_bands):
            early_stopping = EarlyStopping(monitor='Class_Acc',
                                           **self.callbacks_args
                                           )

            self.callbacks[b] = tf.keras.callbacks.CallbackList([
                early_stopping,
                # profiler,
            ],
                model=self.models[b],
            )

    def run_test(self, path_parameters, path_records_test, path_weights, df_paths=None):
        # Read the parameters
        self.load_setup(path_parameters)
        # Load scalers
        if 'regression' in self.mode:
            # Load the scalers
            with open(self.path_scalers, 'rb') as file:
                self.scalers = pickle.load(file)
        # Initialize dataset
        self.__initialize_dataset_test(path_records_test)
        # Define the input shapes
        self.__define_inputs_test()
        # Add placeholders
        self.__add_placeholders()
        # Build the models
        self.__add_models()
        # Load weights
        self.load_weights(path_weights)
        # Evaluate on the test set
        self.test_loop(print_report=self.print_report)
        # Save the results if chosen
        self.save_results(df_paths)
    def run_test_test(self, path_parameters, path_records_test, path_weights, df_paths=None):
        # Read the parameters
        self.load_setup(path_parameters)
        # Load scalers
        if 'regression' in self.mode:
            # Load the scalers
            with open(self.path_scalers, 'rb') as file:
                self.scalers = pickle.load(file)
        # Initialize dataset
        self.__initialize_dataset_test(path_records_test)
        # Define the input shapes
        self.__define_inputs_test()
        # Add placeholders
        self.__add_placeholders()
        # Build the models
        self.__add_models()
        # Load weights
        self.load_weights(path_weights)
        # # Evaluate on the test set
        # self.test_loop(print_report=self.print_report)
        # # Save the results if chosen
        # self.save_results(df_paths)
    def load_setup(self, path):
        with open(path) as f:
            all_metadata = json.load(f)
        self.size_hidden_bands = all_metadata['hidden_size_bands']
        self.size_hidden_central = all_metadata['hidden_size_central']
        self.rnn_layers_bands = all_metadata['rnn_layers_bands']
        self.rnn_layers_central = all_metadata['rnn_layers_central']
        self.fc_layers_bands = all_metadata['fc_layers_bands']
        self.fc_layers_central = all_metadata['fc_layers_central']
        self.regression_size = all_metadata['regression_size']
        self.buffer_size = all_metadata['buffer_size']
        self.epochs = all_metadata['epochs']
        self.num_cores = all_metadata['num_cores']
        self.batch_size = all_metadata['batch_size']
        self.dropout = all_metadata['dropout']
        self.lr_bands = all_metadata['lr_bands']
        self.lr_central = all_metadata['lr_central']
        self.val_steps = all_metadata['val_steps']
        self.max_to_keep = all_metadata['max_to_keep']
        self.w = all_metadata['w']
        self.s = all_metadata['s']
        self.num_classes = all_metadata['num_classes']
        self.n_bands = all_metadata['n_bands']
        self.physical_params = all_metadata['Physical_parameters']
        self.mode = all_metadata['mode']
        self.use_output_layers_bands = all_metadata['use_output_bands']
        self.use_output_layers_central = all_metadata['use_output_central']
        self.use_common_layers = all_metadata['use_common_layers']
        self.bidirectional_central = all_metadata['bidirectional_central']
        self.bidirectional_band = all_metadata['bidirectional_band']
        self.layer_norm_params = all_metadata['layer_norm_params']
        self.use_gated_common = all_metadata['use_gated_common']
        self.l1 = all_metadata['l1']
        self.l2 = all_metadata['l2']
        self.steps_wait = all_metadata['steps_wait']
        self.use_class_weights = all_metadata['use_class_weights']
        self.numpy_weights = np.array(all_metadata['numpy_weights'])
        self.class_weights = all_metadata['class_weights']
        self.vector_weights = tf.constant(self.numpy_weights, dtype=tf.float32)
        self.max_l = all_metadata['max_l']
        self.min_l = all_metadata['min_l']
        self.max_N = all_metadata['max_n']
        self.min_n = all_metadata['min_n']
        self.N_skip = all_metadata['N_skip']
        self.use_raw_input_central = all_metadata['use_raw_input_central']
        self.train_steps_central = all_metadata['train_steps_central']
        self.print_report = all_metadata['print_report']
        self.path_scalers = all_metadata['path_scalers']
        self.loss_weights_central = all_metadata['loss_weights_central']
        self.callbacks_args = all_metadata['callbacks_args']

        self.element_class = all_metadata['element_class']
        self.trans = all_metadata['trans']
        # Convert keys to int '0':'RRab' to 0:'RRab'
        self.trans = {int(key): self.trans[key] for key in self.trans.keys()}
        self.trans_inv = all_metadata['trans_inv']

    def save_setup(self):
        """Save the dictionaries and metadata's to reconstruct the model for test."""

        self.all_metadata = {}
        self.all_metadata['hidden_size_bands'] = self.size_hidden_bands
        self.all_metadata['hidden_size_central'] = self.size_hidden_central
        self.all_metadata['rnn_layers_bands'] = self.rnn_layers_bands
        self.all_metadata['rnn_layers_central'] = self.rnn_layers_central
        self.all_metadata['fc_layers_bands'] = self.fc_layers_bands
        self.all_metadata['fc_layers_central'] = self.fc_layers_central
        self.all_metadata['regression_size'] = self.regression_size
        self.all_metadata['buffer_size'] = self.buffer_size
        self.all_metadata['epochs'] = self.epochs
        self.all_metadata['num_cores'] = self.num_cores
        self.all_metadata['batch_size'] = self.batch_size
        self.all_metadata['dropout'] = self.dropout
        self.all_metadata['lr_bands'] = self.lr_bands
        self.all_metadata['lr_central'] = self.lr_central
        self.all_metadata['val_steps'] = self.val_steps
        self.all_metadata['max_to_keep'] = self.max_to_keep
        self.all_metadata['w'] = self.w
        self.all_metadata['s'] = self.s
        self.all_metadata['num_classes'] = self.num_classes
        self.all_metadata['n_bands'] = self.n_bands
        self.all_metadata['Physical_parameters'] = self.physical_params
        self.all_metadata['mode'] = self.mode

        self.all_metadata['use_output_bands'] = self.use_output_layers_bands
        self.all_metadata['use_output_central'] = self.use_output_layers_central
        self.all_metadata['use_common_layers'] = self.use_common_layers
        self.all_metadata['bidirectional_central'] = self.bidirectional_central
        self.all_metadata['bidirectional_band'] = self.bidirectional_band
        self.all_metadata['layer_norm_params'] = self.layer_norm_params
        self.all_metadata['use_gated_common'] = self.use_gated_common
        self.all_metadata['l1'] = self.l1
        self.all_metadata['l2'] = self.l2
        self.all_metadata['steps_wait'] = self.steps_wait
        self.all_metadata['use_class_weights'] = self.use_class_weights

        self.all_metadata['numpy_weights'] = list(self.numpy_weights)
        self.all_metadata['class_weights'] = self.class_weights

        self.all_metadata['max_l'] = self.max_l
        self.all_metadata['min_l'] = self.min_l
        self.all_metadata['max_n'] = self.max_N
        self.all_metadata['min_n'] = self.min_n

        self.all_metadata['N_skip'] = self.N_skip
        self.all_metadata['use_raw_input_central'] = self.use_raw_input_central
        self.all_metadata['train_steps_central'] = self.train_steps_central
        self.all_metadata['print_report'] = self.print_report
        self.all_metadata['path_scalers'] = self.path_scalers
        self.all_metadata['loss_weights_central'] = self.loss_weights_central
        self.all_metadata['callbacks_args'] = self.callbacks_args

        self.all_metadata['trans'] = self.trans
        self.all_metadata['trans_inv'] = self.trans_inv
        self.all_metadata['element_class'] = self.element_class

        class_keys = self.trans
        keys = [str(k) for k in class_keys.keys()]
        class_keys = dict(zip(keys, class_keys.values()))
        self.all_metadata['class_keys'] = class_keys

        path = os.path.join(self.model_dir, 'all_settings.json')
        with open(path, 'w') as fp:
            json.dump(self.all_metadata, fp)

    def save_weights(self):
        # Save the weights of the scapulars and spine
        for b in range(self.n_bands):
            bb = str(b)
            self.models[b].save_weights(self.model_dir + '/model_' + bb)
        self.model_central.save_weights(self.model_dir + '/model_central')

    def load_weights(self, models_path):
        for b in range(self.n_bands):
            bb = str(b)
            self.models[b].load_weights(models_path + '/model_' + bb)
        self.model_central.load_weights(models_path + '/model_central')

    def __initialize_dataset_test(self, filename_test):
        loader = Parser.Parser(physical_parameters=self.physical_params,
                               n_bands=self.n_bands,
                               num_classes=self.num_classes,
                               w=self.w,
                               batch_size=self.batch_size,
                               num_threads=self.num_cores,
                               buffer_size=self.buffer_size,
                               mode=self.mode,
                               )
        print(filename_test)
        self.dataset_test = loader.get_dataset(filename=filename_test,
                                               epochs=1,
                                               shuffle=False
                                               )

    def __initialize_datasets(self, filename_train, filename_val, filename_test):
        loader = Parser.Parser(physical_parameters=self.physical_params,
                               n_bands=self.n_bands,
                               num_classes=self.num_classes,
                               w=self.w,
                               batch_size=self.batch_size,
                               num_threads=self.num_cores,
                               buffer_size=self.buffer_size,
                               mode=self.mode,
                               )
        self.dataset_train = loader.get_dataset(filename=filename_train,
                                                # epochs= self.epochs,
                                                epochs=1,
                                                shuffle=True
                                                )
        self.dataset_val = loader.get_dataset(filename=filename_val,
                                              epochs=1,
                                              shuffle=False,
                                              )
        self.dataset_test = loader.get_dataset(filename=filename_test,
                                               epochs=1,
                                               shuffle=False
                                               )

    @staticmethod
    def __get_optim(lr, optimizer='Adam'):
        # Specify the scheduler
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr,
                                                                     decay_steps=60,
                                                                     decay_rate=0.95,
                                                                     staircase=False)
        # Specify which optimizer to use
        if optimizer == 'AdamW':
            optim = tf.keras.optimizers.AdamW(learning_rate=lr_schedule,
                                         weight_decay=1e-4,
                                         )
        else:
            optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        return optim

    @staticmethod
    def __last_relevant(output, length):
        """Get the last relevant output from the network"""
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = tf.cast(output.get_shape()[2], tf.int32)

        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant
