import matching
import data_loader


def preprocess(experiment):
    n_data_files = experiment.n_evaluations

    for i in range(n_data_files):
        iterators, meta = data_loader.load_data(experiment, i, batch_size=[None],
              splits=[1], do_shuffle=[False])
        yield #matching ()