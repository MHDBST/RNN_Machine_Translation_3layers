from __future__ import print_function

import logging
import numpy
import os
import pickle

from theano import printing as pr
from theano import pp
from blocks import initialization as ini
from theano import pprint
from collections import Counter
from theano import tensor
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from main_loop import MainLoop
from blocks.model import Model
# from search import BeamSearch
from mySearch import BeamSearch
from blocks.select import Selector

# from blocks.serialization import load_parameter_values

from checkpoint import CheckpointNMT, LoadNMT
from model import BidirectionalEncoder, Decoder
from sampling import BleuValidator, Sampler, SamplingBase
from stream import (get_tr_stream, get_dev_stream,_ensure_special_tokens)

from toolz import merge

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout


try:
    from blocks.extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False



logger = logging.getLogger(__name__)


# extension ha ye tabe b name do daran. ina dar heine ejra b suratet tanvaobi ba tavajoh b sharayet ejra mishan. mitunim
# mitunim tu codesh taghirati bedimom har bar ejda bebinim

def main(mode, config, use_bokeh=False):







     # Construct model
     logger.info('Building RNN encoder-decoder')
     encoder = BidirectionalEncoder(
         config['src_vocab_size'], config['enc_embed'], config['enc_nhids'])

     decoder = Decoder(
         config['trg_vocab_size'], config['dec_embed'], config['dec_nhids'],
         config['enc_nhids'] * 3)
     print(mode)

     if mode == "train":
        
        print(mode)


        # Create Theano variables
        logger.info('Creating theano variables')
        source_sentence = tensor.lmatrix('source')
        source_sentence_mask = tensor.matrix('source_mask')

        target_sentence = tensor.lmatrix('target')
        target_sentence_mask = tensor.matrix('target_mask')
        sampling_input = tensor.lmatrix('input')


        # creating reorder variables
        source_sentence_reorder = tensor.lmatrix('source_reorder')
        source_sentence_reorder_mask = tensor.matrix('source_reorder_mask')
        sampling_input_reorder = tensor.lmatrix('input_reorder')

        # Get training and development set streams

        # Get training and development set streams
        tr_stream = get_tr_stream(**config)


        dev_stream = get_dev_stream(**config)

        # Get cost of the model
        cost = decoder.cost(
            encoder.apply(source_sentence, source_sentence_mask ,source_sentence_reorder, source_sentence_reorder_mask),
            source_sentence_mask, target_sentence, target_sentence_mask)

        logger.info('Creating computational graph')
        cg = ComputationGraph(cost)
        # print('cg ' ,  cg)

        # Initialize model
        logger.info('Initializing model')
        # encoder.weights_init = decoder.weights_init = [0]
        encoder.weights_init = decoder.weights_init = IsotropicGaussian(
            config['weight_scale'])
        encoder.biases_init = decoder.biases_init = Constant(0)
        encoder.push_initialization_config()
        decoder.push_initialization_config()
        encoder.bidir.prototype.weights_init = Orthogonal()
        decoder.transition.weights_init = Orthogonal()
        encoder.initialize()
        decoder.initialize()



        # apply dropout for regularization
        if config['dropout'] < 1.0:
            # dropout is applied to the output of maxout in ghog
            logger.info('Applying dropout')
            dropout_inputs = [x for x in cg.intermediary_variables
                              if x.name == 'maxout_apply_output']
            cg = apply_dropout(cg, dropout_inputs, config['dropout'])

        # Apply weight noise for regularization
        if config['weight_noise_ff'] > 0.0:
            logger.info('Applying weight noise to ff layers')
            enc_params = Selector(encoder.lookup).get_params().values()
            enc_params += Selector(encoder.fwd_fork).get_params().values()
            enc_params += Selector(encoder.back_fork).get_params().values()
            dec_params = Selector(
                decoder.sequence_generator.readout).get_params().values()
            dec_params += Selector(
                decoder.sequence_generator.fork).get_params().values()
            dec_params += Selector(decoder.state_init).get_params().values()
            cg = apply_noise(
                cg, enc_params+dec_params, config['weight_noise_ff'])

        # Print shapes
        shapes = [param.get_value().shape for param in cg.parameters]
        logger.info("Parameter shapes: ")
        for shape, count in Counter(shapes).most_common():
            logger.info('    {:15}: {}'.format(shape, count))
        logger.info("Total number of parameters: {}".format(len(shapes)))

        # Print parameter names
        enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                                   Selector(decoder).get_parameters())
        logger.info("Parameter names: ")
        for name, value in enc_dec_param_dict.items():
            logger.info('    {:15}: {}'.format(value.get_value().shape, name))
        logger.info("Total number of parameters: {}"
                    .format(len(enc_dec_param_dict)))

        # Set up training model
        logger.info("Building model")
        training_model = Model(cost)

        # Set extensions
        logger.info("Initializing extensions")
        extensions = [
            FinishAfter(after_n_batches=config['finish_after']),
            TrainingDataMonitoring([cost], after_batch=True),
            Printing(after_batch=True),
            CheckpointNMT(config['saveto'],
                          every_n_batches=config['save_freq'])
        ]


        # Set up beam search and sampling computation graphs if necessary
        if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
            logger.info("Building sampling model")

            sampling_representation = encoder.apply(
                    sampling_input, tensor.ones(sampling_input.shape),sampling_input_reorder, tensor.ones(sampling_input_reorder.shape))
            generated = decoder.generate(
                    sampling_input, sampling_representation)
            search_model = Model(generated)
            # print(ComputationGraph(generated[1]).variables)
            # for var in ComputationGraph(generated[1]).variables:
            #     if hasattr(var.tag, 'name'):
            #         print(var.tag.name)

            # fek konam moteghayerhaie k mikhad tu model bashano dare kam mikone
            _, samples = VariableFilter(
                bricks=[decoder.sequence_generator], name="outputs")(
                    ComputationGraph(generated[1]))


        # Add sampling


        # self, model, data_stream, hook_samples=1,
        #          src_vocab=None, trg_vocab=None, src_ivocab=None,
                 # trg_ivocab=None, src_vocab_size=None, **kwargs
        if config['hook_samples'] >= 1:
            logger.info("Building sampler")
            extensions.append(
                Sampler(model=search_model, data_stream=tr_stream,
                        hook_samples=config['hook_samples'],

                        every_n_batches=config['sampling_freq'],
                        src_vocab_size=config['src_vocab_size']))

        # Add early stopping based on bleu
        if config['bleu_script'] is not None:
            logger.info("Building bleu validator")
            extensions.append(
                BleuValidator(sampling_input,sampling_input_reorder, samples=samples, config=config,
                              model=search_model, data_stream=dev_stream,
                              normalize=config['normalized_bleu'],
                              every_n_batches=config['bleu_val_freq']))

        # Reload model if necessary
        if config['reload']:
            extensions.append(LoadNMT(config['saveto']))

        # Plot cost in bokeh if necessary
        if use_bokeh and BOKEH_AVAILABLE:
            extensions.append(
                Plot('Fa-Reorder', channels=[['decoder_cost_cost']],
                     after_batch=True))

        # Set up training algorithm
        logger.info("Initializing training algorithm")
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()]),on_unused_sources='warn'
        )

        # Initialize main loop
        logger.info("Initializing main loop")
        main_loop = MainLoop(
            model=training_model,
            algorithm=algorithm,
            data_stream=tr_stream,
            extensions=extensions
        )

        # Train!
        main_loop.run()

     elif mode == 'translate':

        # Create Theano variables
        logger.info('Creating theano variables')
        sampling_input = tensor.lmatrix('source')
        sampling_input_reorder = tensor.lmatrix('source_reorder')

        # Get test set stream
        test_stream = get_dev_stream(
            config['test_set'], config['test_set_reorder'],config['src_vocab'],
            config['src_vocab_size'], config['unk_id'])
        ftrans = open(config['test_set'] + '.trans.out', 'w')

        # Helper utilities
        sutils = SamplingBase()
        unk_idx = config['unk_id']
        src_eos_idx = config['src_vocab_size'] - 1

        trg_eos_idx = config['trg_vocab_size'] - 1

        # Get beam search
        logger.info("Building sampling model")
        sampling_representation = encoder.apply(
            sampling_input, tensor.ones(sampling_input.shape), sampling_input_reorder,
            tensor.ones(sampling_input_reorder.shape))
        generated = decoder.generate(sampling_input, sampling_representation)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs
        beam_search = BeamSearch(samples=samples)


        logger.info("Loading the model..")
        model = Model(generated)
        loader = LoadNMT(config['saveto'])
        loader.set_model_parameters(model, loader.load_parameters())

        # Get target vocabulary
        trg_vocab = _ensure_special_tokens(
            pickle.load(open(config['trg_vocab'])), bos_idx=0,
            eos_idx=trg_eos_idx, unk_idx=unk_idx)
        trg_ivocab = {v: k for k, v in trg_vocab.items()}

        print("Started translation:")

        logger.info("Started translation: ")
        total_cost = 0.0

        # for i, line in enumerate(test_stream.get_epoch_iterator()):
        for i, (line, line_reorder) in enumerate(
                zip(test_stream[0].get_epoch_iterator(), test_stream[1].get_epoch_iterator())):
            print(i)
            if (i % 100 == 0) : 
                print("i is " , i)

            seq = sutils._oov_to_unk(
                line[0], config['src_vocab_size'], unk_idx)
            input_ = numpy.tile(seq, (config['beam_size'], 1))

            seq_reorder = sutils._oov_to_unk(
                line_reorder[0], config['src_vocab_size'], unk_idx)
            input_reorder_ = numpy.tile(seq_reorder, (config['beam_size'], 1))

            inputs = {sampling_input: input_, sampling_input_reorder: input_reorder_}

            # print("inputs" , inputs)

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                beam_search.search(
                input_values=inputs,
                max_length=3 * len(seq), eol_symbol=src_eos_idx,
                ignore_first_eol=True)
                # beam_search.search(
                #     input_values={sampling_input: input_},
                #
                #     max_length=3*len(seq), eol_symbol=src_eos_idx,ignore_first_eol=True)


            # trans, costs = \
            #     beam_search.search(
            #         input_values={sampling_input: input_2},
            #         trg_ivocab=trg_ivocab,
            #         source_sentence=sampling_input,
            #         max_length=3 * len(seq), eol_symbol=trg_eos_idx,
            #         ignore_first_eol=True)

            # normalize costs according to the sequence lengths
            if config['normalized_bleu']:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            best = numpy.argsort(costs)[0]
            try:
                total_cost += costs[best]
                trans_out = trans[best]

                # convert idx to words
                trans_out = sutils._idx_to_word(trans_out, trg_ivocab)

                # REMOVE </S> end of file
                splits = trans_out.split()
                if (splits[len(splits) - 1]) == "</S>":
                    splits = splits[0:len(splits) - 1]
                trans_out = " ".join([word for word in splits])
                # print(trans_out)

            except ValueError:
                logger.info(
                    "Can NOT find a translation for line: {}".format(i+1))
                trans_out = '<UNK>'

            print(trans_out, file=ftrans)

            if i != 0 and i % 100 == 0:
                logger.info(
                    "Translated {} lines of test set...".format(i))
        logger.info("Total cost of the test: {}".format(total_cost))
        ftrans.close()
