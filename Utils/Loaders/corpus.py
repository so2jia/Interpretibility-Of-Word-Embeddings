def read(input_file, max_words=-1):
    """
    Reads in corpus
    Parameters
    ----------
    input_file Input file
    max_words Maximum words to read

    Returns
    -------
    List of the words
    """
    with open(input_file, mode='r') as file:
        words = file.read().splitlines()
        return words[0: max_words]