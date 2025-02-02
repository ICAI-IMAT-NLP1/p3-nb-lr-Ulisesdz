import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.size(1) # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples
        class_priors: Dict[int, torch.Tensor] = {}
        class_counts = Counter(labels.tolist())
        total_samples = labels.size(0)

        for class_label, count in class_counts.items():
            class_priors[class_label] = torch.tensor(count/total_samples)
        
        return class_priors


    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # TODO: Estimate conditional probabilities for the words in features and apply smoothing
        class_word_counts: Dict[int, torch.Tensor] = {}
        class_total_word_counts: Dict[int, int] = {}
        class_labels = labels.tolist()

        # Creo vector de ceros para cada clase y el total de palabras
        for class_label in set(class_labels):
            class_word_counts[int(class_label)] = torch.zeros(self.vocab_size) 
            class_total_word_counts[int(class_label)] = 0

        # Acumulo el número de veces que aparece cada palabre para cada clase y el total
        for i, class_label in enumerate(class_labels):
            class_word_counts[int(class_label)] += features[i]
            class_total_word_counts[int(class_label)] += features[i].sum().item()
        
        # Aplico Laplace Smoothing y calculo las probabilidades condicionales
        for class_label in class_word_counts:
            word_counts = class_word_counts[class_label]
            total_class_words = class_total_word_counts[class_label]

            # Actualizo el vector de cada clase a probabilidad aplicando el Smoothing
            class_word_counts[class_label] = (word_counts + delta) / (total_class_words + delta * self.vocab_size)

        return class_word_counts
    

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words
        log_posteriors: torch.Tensor = torch.zeros(len(self.class_priors))

        for class_label in range(len(self.class_priors)):
            # Logaritmo de la probabilidad prior para esta clase
            log_prior = torch.log(self.class_priors[class_label])

            log_conditional = 0.0            
            # Suma de  los logaritmos de las probabilidades condicionales de las palabras
            for word_idx in range(feature.size(0)):
                if feature[word_idx] > 0:
                    # Multiplico por cuantas veces aparece cada palabra
                    log_conditional += torch.log(self.conditional_probabilities[class_label][word_idx]*feature[word_idx])

            # Log-posterior para esta clase
            log_posteriors[class_label] = log_prior + log_conditional

        return log_posteriors


    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # TODO: Calculate log posteriors and obtain the class of maximum likelihood 
        log_posteriors = self.estimate_class_posteriors(feature).argmax()
        pred: int = log_posteriors.item()
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        log_posteriors = self.estimate_class_posteriors(feature)
        probs: torch.Tensor = torch.nn.functional.softmax(log_posteriors, dim=0)
        return probs