def batch_generator(trainingData, tdp, batch_size):
    while True:
        batch_data = loadData(trainingData, batch_size)
        yield tdp.preprocess(batch_data)


def loadData(lst, batch_size):
    """  Yields batch of specified size """
    for i in range(0, len(lst), batch_size):
        return lst[i: i + batch_size]
