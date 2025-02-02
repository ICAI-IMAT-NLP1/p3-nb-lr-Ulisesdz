from typing import List, Dict
from collections import Counter
import torch
import re

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # TODO: Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    examples: List[SentimentExample] = []
    with open(infile, "r", encoding='utf-8') as archivo:
        for linea in archivo.readlines():
            # Eliminar las etiquetas HTML (ej. <br />)
            linea_limpia = re.sub(r'<.*?>', '', linea.strip())
            # Buscar la última tabulación
            tab_index = linea_limpia.rfind("\t")

            if tab_index != -1:
                # Dividir en la última tabulación
                content = [linea_limpia[:tab_index], linea_limpia[tab_index + 1:]]
                
                if len(content) == 2:
                    sentence = tokenize(content[0])
                    sentiment_label = int(content[1])  # La etiqueta de sentimiento está después de la última tabulación
                    sentiment_obj = SentimentExample(sentence, sentiment_label)
                    examples.append(sentiment_obj)
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}
    indx = 0
    for sentence in examples:
        for word in sentence.words:
            if word not in vocab:
                vocab[word] = indx
                indx += 1

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full
    bow: torch.Tensor = torch.zeros(len(vocab))
    for word in text:
        if word in vocab:
            if not binary:
                bow[vocab[word]] += 1
            else:
                bow[vocab[word]] = 1
    return bow
