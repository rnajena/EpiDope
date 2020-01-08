import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print('\nLoading packages.')

import os
from keras.backend import set_session
from keras import layers, optimizers, models
from keras.regularizers import l2
import time
from sklearn.preprocessing import LabelEncoder
import numpy as np
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid, Range1d, Label
from bokeh.layouts import column
from bokeh.models.glyphs import Text
from bokeh.models import Legend
from bokeh.plotting import figure, output_file, save
import tensorflow as tf
from epidope.utils import embedder
# silence deprecation warnings
try:
    import tensorflow.python.util.deprecation as deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
except:
    pass

# filters tensor flow output (the higher the number the more ist filtered)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # {0, 1, 2 (warnings), 3 (errors)}

starttime = time.time()


class Protein_seq():
    def __init__(self, sequence, score, over_threshold, positions=None):
        self.sequence = sequence
        self.score = score
        self.over_threshold = over_threshold
        if positions == None:
            self.positions = list(range(1, len(self.sequence) + 1))
        else:
            self.positions = positions


def build_model(nodes, dropout, seq_length, weight_decay_lstm=1e-6, weight_decay_dense=1e-3, non_binary=False,
                own_embedding=False, both_embeddings=False):
    if own_embedding:
        inputs = layers.Input(shape=(seq_length,))
        seq_input = layers.Embedding(27, 10, input_length=seq_length)(inputs)
        hidden = layers.Bidirectional(
            layers.LSTM(nodes, return_sequences=True, dropout=dropout,
                        recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
                        recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(seq_input)
        hidden = layers.Bidirectional(
            layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
                        recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(hidden)

    elif both_embeddings:
        embedding_input = layers.Input(shape=(seq_length, 1024))
        left = layers.Bidirectional(
            layers.LSTM(nodes // 2, input_shape=(seq_length, 1024), return_sequences=True, dropout=dropout,
                        recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
                        recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(
            embedding_input)
        left = layers.Dense(nodes)(left)
        left = layers.LeakyReLU(alpha=0.01)(left)
        out_left = layers.Flatten()(left)

        seq_input = layers.Input(shape=(seq_length,))
        right = layers.Embedding(27, 10, input_length=seq_length)(seq_input)
        right = layers.Bidirectional(
            layers.LSTM(nodes, return_sequences=True, dropout=dropout,
                        recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
                        recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(right)
        right = layers.Dense(nodes)(right)
        right = layers.LeakyReLU(alpha=0.01)(right)
        out_right = layers.Flatten()(right)

        hidden = layers.concatenate([out_left, out_right])

    else:
        inputs = layers.Input(shape=(seq_length, 1024))
        hidden = layers.Bidirectional(
            layers.LSTM(nodes, input_shape=(seq_length, 1024), return_sequences=True, dropout=dropout,
                        recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
                        recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(inputs)
        hidden = layers.Bidirectional(
            layers.LSTM(nodes, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(weight_decay_lstm),
                        recurrent_regularizer=l2(weight_decay_lstm), bias_regularizer=l2(weight_decay_lstm)))(hidden)

    hidden = layers.Dense(nodes, kernel_regularizer=l2(weight_decay_dense), bias_regularizer=l2(weight_decay_dense))(
        hidden)

    hidden = layers.LeakyReLU(alpha=0.01)(hidden)

    out = layers.Dense(2, activation='softmax', kernel_regularizer=l2(weight_decay_dense),
                       bias_regularizer=l2(weight_decay_dense))(hidden)
    if both_embeddings:
        model = models.Model(inputs=[embedding_input, seq_input], outputs=out)
    else:
        model = models.Model(inputs=inputs, outputs=out)

    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    if non_binary:
        model.compile(optimizer="adam", loss='binary_crossentropy')
    else:
        if both_embeddings:
            model.compile(optimizer=adam, loss='binary_crossentropy')
            model.summary()
            return model
        model.compile(optimizer=adam, loss='binary_crossentropy')
    model.summary()
    return model


def split_AA_seq(seq, slicesize, shift):
    """
    Takes input sequence and slicesize: Returns slices of that sequence with a slice length of 'slicesize' with a sliding window of 1.
    """
    splited_AA_seqs = []
    for i in range(0, len(seq) - slicesize):
        splited_AA_seqs.append([i + (slicesize // 2) - shift, seq[i:i + slicesize]])
    return np.array(splited_AA_seqs)


def split_embedding_seq(embeddings, slicesize, shift):
    assert len(
        embeddings) == 1, "splitting of embeddings not intended for multiple proteins (state of affairs 12.06.19)"
    splited_em_seqs = []
    for protein in embeddings:
        splited_em_seq = []
        for i in range(0, len(protein) - slicesize):
            splited_em_seq.append([i + (slicesize // 2) - shift, protein[i:i + slicesize]])
        foo = np.array(splited_em_seq)
        splited_em_seqs.append(splited_em_seq)
    return np.array(splited_em_seqs[0])


def read_fasta(multifasta, delim, idpos):
    """reading input fasta file"""

    fasta = {}
    fastaheader = {}
    print('Reading input fasta.')
    with open(multifasta, 'r') as infile:
        acNumber = ''
        for line in infile:
            if line.startswith('>'):
                if delim:
                    acNumber = line.split(delim)[idpos].strip().strip('>')
                    fastaheader[acNumber] = line.strip()
                else:
                    acNumber = line.split()[idpos].strip().strip('>')
                    fastaheader[acNumber] = line.strip()
            else:
                if acNumber in fasta:
                    fasta[acNumber] += line.strip()
                else:
                    fasta[acNumber] = line.strip()
    return fasta, fastaheader


def read_epi_seqs(epi_seqs):
    ##### reading provided epitope lists #######
    epitopes = list()
    print('Reading provided epitope sequences.')
    with open(epi_seqs, 'r') as infile:
        for line in infile:
            epitopes.append(line.strip())
    print('There were ' + str(len(epitopes)) + ' epitope sequences provided.')
    return epitopes


def read_non_epi_seqs(non_epi_seqs):
    nonepitopes = list()
    print('Reading provided non-epitope sequences.')
    with open(non_epi_seqs, 'r') as infile:
        for line in infile:
            nonepitopes.append(line.strip())
    print('There were ' + str(len(nonepitopes)) + ' non-epitope sequences provided.')
    return nonepitopes


def ensemble_prediction(model, path, inputs_test, suffix, nb_samples, middle_name="", prediction_weights=False,
                        nb_classes=2):
    models_filenames = []
    for file in sorted(os.listdir(path)):
        if file.endswith(f"_{suffix}.hdf5") and file.startswith(f"weights_model_{middle_name}k-fold_run_"):
            models_filenames.append(os.path.join(path, file))

    preds = []
    for fn in models_filenames:
        model.load_weights(fn, by_name=True)
        pred = model.predict(inputs_test)
        preds.append(pred)

    if not prediction_weights:
        prediction_weights = [1. / len(models_filenames)] * len(models_filenames)
    weighted_predictions = np.zeros((nb_samples, nb_classes), dtype='float32')
    for weight, prediction in zip(prediction_weights, preds):
        weighted_predictions += weight * np.array(prediction)

    return weighted_predictions


def predict_files(fasta, slicesize, epitope_threshold, threads):
    # constants
    shift = 24
    local_embedding = False
    use_circular_filling = False
    both_embeddings = True

    ##### progress vars ####
    filecounter = 1
    total = str(len(fasta))

    def make_model_and_embedder(slicesize, threads):
        print('Deep Neural Network model summary:')
        nodes = 10
        model = build_model(nodes, dropout=0, seq_length=slicesize, both_embeddings=both_embeddings)
        elmo_embedder = embedder.Elmo_embedder(threads=threads)
        return model, elmo_embedder

    def show_progress(filecounter, printlen=1):

        ############### progress ###############
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - starttime))
        printstring = f'Predicting scores for: {geneid}    File: {filecounter} / {total}   Elapsed time: {elapsed_time}'
        if len(printstring) < printlen:
            print(' ' * printlen, end='\r')
        print(printstring, end='\r')
        filecounter += 1
        return filecounter

    def prepare_input(protein, elmo_embedder, local_embedding, both_embeddings, use_circular_filling, shift, slicesize):
        if local_embedding:
            warnings.warn("local_embedding is deprecated")
            exit()
            # seq = protein.upper()
            #
            # if use_circular_filling:
            #     seq_extended = list(seq[-shift:] + seq + seq[0:shift])
            # else:
            #     seq_extended = np.array(["-"] * (len(seq) + (shift * 2)))
            #     seq_extended[shift:-shift] = np.array(list(seq))
            #
            # seq_slices = split_AA_seq(seq_extended, slicesize, shift)
            # positions = seq_slices[:, 0]
            # seq_slices_input = np.array([list(i) for i in seq_slices[:, 1]])
            # X_test = elmo_embedder.elmo_embedding(seq_slices_input, 0, slicesize)
            # nb_samples = X_test.shape[0]

        # embedding whole protein version
        else:
            seq = np.array([list(protein.upper())])

            X_test = elmo_embedder.elmo_embedding(seq)

            seq_extended = np.zeros((1, len(protein) + (shift * 2), 1024), dtype=np.float32)
            if use_circular_filling:
                seq_extended[0, 0:shift] = X_test[0, -shift:]
                seq_extended[0, -shift:] = X_test[0, 0:shift]
            seq_extended[0, shift:-shift] = X_test[0]
            seq_slices = split_embedding_seq(np.array(seq_extended), slicesize, shift)
            positions = seq_slices[:, 0]
            X_test = np.stack(seq_slices[:, 1])
            nb_samples = X_test.shape[0]

            if both_embeddings:
                seq_extended = ["-" * shift + protein.upper() + "-" * shift]

                amino = "ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
                encoder = LabelEncoder()
                encoder.fit(list(amino))

                seq_transformed = np.array(
                    list(map(encoder.transform, np.array([list(i.upper()) for i in seq_extended]))))

                seq_slices = split_AA_seq(seq_transformed[0], slicesize, shift)
                X_test_seq = np.stack(seq_slices[:, 1])
                X_test = [X_test, X_test_seq]

        return X_test, nb_samples, positions

    def predict_protein(model, X_test, protein, nb_samples, positions, epitope_threshold, threads):
        path_weights = "/home/go96bix/projects/epitop_pred/data_generator_bepipred_binary_double_cluster_0.8_0.5_seqID"
        suffix_weights = "both_embeddings_50epochs"
        middle_name = ""

        Y_pred_test = ensemble_prediction(model, path=path_weights, nb_samples=nb_samples, middle_name=middle_name,
                                          inputs_test=X_test, suffix=suffix_weights)

        # the column 0 in Y_pred_test is the likelihood that the slice is NOT a Epitope, for us mostly interesting
        # is col 1 which contain the likelihood of being a epitope
        epi_score = Y_pred_test[:, 1]

        # use leading and ending zeros so that the score array has the same length as the input sequence
        score = np.zeros(len(protein))
        # leading AAs which are not predictable get value of first predicted value (were this AA where involved)
        score[0:int(positions[0])] = epi_score[0]
        # last AAs which are not predictable get value of last predicted value (were this AA where involved)
        score[int(positions[-1]):] = epi_score[-1]
        score[np.array(positions, dtype=int)] = epi_score
        score_bool = score > epitope_threshold

        protein_result = Protein_seq(sequence=protein, score=score, over_threshold=score_bool)
        return protein_result

    # make_model
    model, elmo_embedder = make_model_and_embedder(slicesize, threads)

    print('\nPredicting DeEpiPred scores.')

    # set number of threads used
    if threads != 1000:
        config = tf.ConfigProto(intra_op_parallelism_threads=threads, inter_op_parallelism_threads=threads)
        session = tf.Session(config=config)
        set_session(session)

    protein_results_dict = {}
    for geneid in fasta:
        protein = fasta[geneid]

        filecounter = show_progress(filecounter)

        X_test, nb_samples, positions = prepare_input(protein, elmo_embedder, local_embedding, both_embeddings,
                                                      use_circular_filling, shift, slicesize)

        protein_result = predict_protein(model, X_test, protein, nb_samples, positions, epitope_threshold, threads)
        protein_results_dict.update({geneid: protein_result})

    return protein_results_dict


def output_results(outdir, protein_results_dict, epitope_threshold, epitope_slicelen, slice_shiftsize):
    ##############################################
    ############### Output results ###############
    ##############################################

    if not os.path.exists(outdir + '/epidope'):
        os.makedirs(outdir + '/epidope')

    ######## epitope table #########
    predicted_epitopes = {}

    print(
        f'\n\nWriting predicted epitopes to:\n{outdir}/predicted_epitopes.csv\n{outdir}/predicted_epitopes_sliced.faa')

    with open(f'{outdir}/predicted_epitopes.csv', 'w') as outfile:
        with open(f'{outdir}/predicted_epitopes_sliced.faa', 'w') as outfile2:
            with open(f'{outdir}/epidope_scores.csv', 'w') as outfile3:
                outfile.write('#Gene_ID\tstart\tend\tsequence\tscore')
                outfile3.write(f'position/header\taminoacid\tscore\n')

                for geneid in protein_results_dict:
                    scores = protein_results_dict[geneid].score
                    seq = protein_results_dict[geneid].sequence
                    predicted_epis = set()
                    predicted_epitopes[geneid] = []
                    newepi = True
                    start = 0
                    end = 0
                    i = 0
                    out = f'{outdir}/epidope/{geneid}.csv'
                    with open(out, 'w') as outfile6:
                        # write complete scores to file
                        outfile6.write('#Aminoacid\tDeepipred\n')
                        outfile3.write(f'>{geneid}\n')
                        for x in range(len(seq)):
                            outfile3.write(f'{x+1}\t{seq[x]}\t{scores[x]}\n')
                            outfile6.write(f'{x+1}\t{seq[x]}\t{scores[x]}\n')
                        for score in scores:
                            if score >= epitope_threshold:
                                if newepi:
                                    start = i
                                    newepi = False
                                else:
                                    end = i
                            else:
                                newepi = True
                                if end - start >= 8:
                                    predicted_epis.add(
                                        (start + 1, end + 1, seq[start:end + 1], np.mean(scores[start:end + 1])))
                            i += 1
                        if end - start >= 8:
                            predicted_epis.add(
                                (start + 1, end + 1, seq[start:end + 1], np.mean(scores[start:end + 1])))
                        predicted_epis = sorted(predicted_epis)
                        epiout = ''
                        for epi in predicted_epis:
                            epiout = f'{epiout}\n{geneid}\t{epi[0]}\t{epi[1]}\t{epi[2]}\t{epi[3]}'
                            # predicted_epitopes[geneid].append(epi[2])
                            predicted_epitopes[geneid].append(epi)
                            # print slices to blast table
                            ### sliced epitope regions
                            if len(epi[2]) > epitope_slicelen:
                                for i in range(0, len(epi[2]) - (epitope_slicelen - 1), slice_shiftsize):
                                    outfile2.write(
                                        f'>{geneid}|pos_{i+epi[0]}:{i+epi[0]+epitope_slicelen}\n{epi[2][i:i+epitope_slicelen]}\n')
                        outfile.write(f'{epiout}')
    return predicted_epitopes


def plot_results(fasta, protein_results_dict, outdir, epitope_threshold, epitopes, nonepitopes, fastaheader,
                 predicted_epitopes):
    ######## Plots #########
    print('\nPlotting.')

    ##### progress vars ####
    filecounter = 1
    printlen = 1
    total = str(len(fasta))
    ########################

    for geneid in protein_results_dict:

        ############### progress ###############
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - starttime))
        printstring = f'Plotting: {geneid}    File: {filecounter} / {total}   Elapsed time: {elapsed_time}'
        if len(printstring) < printlen:
            print(' ' * printlen, end='\r')
        print(printstring, end='\r')
        printlen = len(printstring)
        filecounter += 1
        #######################################

        # make output dir and create output filename
        if not os.path.exists(outdir + '/plots'):
            os.makedirs(outdir + '/plots')
        out = f'{outdir}/plots/{geneid}.html'
        output_file(out)

        seq = protein_results_dict[geneid].sequence
        pos = protein_results_dict[geneid].positions
        score = protein_results_dict[geneid].score
        protlen = len(seq)

        # create a new plot with a title and axis labels
        p = figure(title=fastaheader[geneid][1:], y_range=(-0.03, 1.03), y_axis_label='Scores', plot_width=1200,
                   plot_height=460, tools='xpan,xwheel_zoom,reset', toolbar_location='above')
        p.min_border_left = 80

        # add a line renderer with legend and line thickness
        l1 = p.line(range(1, protlen + 1), score, line_width=1, color='black', visible=True)
        l2 = p.line(range(1, protlen + 1), ([epitope_threshold] * protlen), line_width=1, color='red', visible=True)

        legend = Legend(items=[('EpiDope', [l1]),
                               ('epitope_threshold', [l2])])

        p.add_layout(legend, 'right')
        p.xaxis.visible = False
        p.legend.click_policy = "hide"

        p.x_range.bounds = (-50, protlen + 51)

        ### plot for sequence
        # symbol based plot stuff

        plot = Plot(title=None, x_range=p.x_range, y_range=Range1d(0, 9), plot_width=1200, plot_height=50, min_border=0,
                    toolbar_location=None)

        y = [1] * protlen
        source = ColumnDataSource(dict(x=list(pos), y=y, text=list(seq)))
        glyph = Text(x="x", y="y", text="text", text_color='black', text_font_size='8pt')
        plot.add_glyph(source, glyph)
        label = Label(x=-80, y=y[1], x_units='screen', y_units='data', text='Sequence', render_mode='css',
                      background_fill_color='white', background_fill_alpha=1.0)
        plot.add_layout(label)

        xaxis = LinearAxis()
        plot.add_layout(xaxis, 'below')
        plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))

        # add predicted epitope boxes
        if predicted_epitopes[geneid]:
            for epi in predicted_epitopes[geneid]:
                start = epi[0]
                end = epi[1]+1
                non_epitope = [-0.02] * (start - 1) + [1.02] * (end-start) + [-0.02] * (
                    (protlen - (start - 1) - (end-start)))
                p.vbar(x=list(pos), bottom=-0.02, top=non_epitope, width=1, alpha=0.2, line_alpha=0,
                       color='darkgreen',
                       legend_label='predicted_epitopes', visible=True)

        # add known epitope boxes
        if epitopes:
            for epi in epitopes:
                hits = [i for i in range(len(seq)) if seq[i:].startswith(epi)]
                if len(hits)>0:
                    for hit in hits:
                        start = hit
                        end = start + len(epi)
                        epitope = [-0.02] * (start) + [1.02] * len(epi) + [-0.02] * ((protlen - (start) - len(epi)))
                        p.vbar(x=list(pos), bottom=-0.02, top=epitope, width=1, alpha=0.2, line_alpha=0, color='blue',
                               legend_label='provided_epitope', visible=True)

        # add non-epitope boxes
        if nonepitopes:
            for epi in nonepitopes:
                hits = [i for i in range(len(seq)) if seq[i:].startswith(epi)]
                if len(hits) > 0:
                    for hit in hits:
                        start = hit
                        end = start + len(epi)
                        non_epitope = [-0.02] * (start) + [1.02] * len(epi) + [-0.02] * (
                            (protlen - (start) - len(epi)))
                        p.vbar(x=list(pos), bottom=-0.02, top=non_epitope, width=1, alpha=0.2, line_alpha=0,
                               color='darkred', legend_label='provided_non_epitope', visible=True)

        column(p, plot)
        save(column(p, plot))
    print()


def start_pipeline(multifasta, outdir, delim, idpos, epitope_threshold, epitope_slicelen, slice_shiftsize, threads,
                   epi_seqs, non_epi_seqs, slicesize=49):
    """
    load input, predict proteins, save results
    :param multifasta: Multi- or Singe- Fasta file with protein sequences.
    :param outdir: Specifies output directory. Default = .
    :param delim: Delimiter char for fasta header. Default = White space.
    :param idpos: Position of gene ID in fasta header. Zero based. Default = 0.
    :param epitope_threshold: Threshold for epitope score. Default = 0.818.
    :param epitope_slicelen: Length of the sliced predicted epitopes. Default = 15.
    :param slice_shiftsize: Shiftsize of the slices on predited epitopes. Default = 5.
    :param threads: Number of processes used for predictions. Default = #CPU-cores.
    :param epi_seqs: File containing a list of known epitope sequences for plotting.
    :param non_epi_seqs: File containing a list of non epitope sequences for plotting.
    :param slicesize: Size of the window used for the epitope prediction.
    :return:
    """

    print(f'The epitope threshold is set to: {epitope_threshold}\n')
    # epitope_threshold = 0.8179208676020304
    # this is representing 15% recall with an precision of 0.6345

    # prepare files
    fasta, fastaheader = read_fasta(multifasta, delim, idpos)

    if epi_seqs is not None:
        epi_seqs = read_epi_seqs(epi_seqs)
    if non_epi_seqs is not None:
        non_epi_seqs = read_non_epi_seqs(non_epi_seqs)

    # calc prediction
    protein_results_dict = predict_files(fasta, slicesize, epitope_threshold, threads)

    # save output
    predicted_epitopes = output_results(outdir, protein_results_dict, epitope_threshold, epitope_slicelen,
                                        slice_shiftsize)

    # plot output
    plot_results(fasta, protein_results_dict, outdir, epitope_threshold, epi_seqs, non_epi_seqs, fastaheader,
                 predicted_epitopes)


if __name__ == '__main__':
    pass
