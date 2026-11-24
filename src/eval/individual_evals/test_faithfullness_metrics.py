from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv

# load the api keys
load_dotenv()

rag_input = "Explain how the K-Nearest Neighbors algorithm works, icluding how the value of K affects performace."

actual_output = """k-Nearest Neighbors is a non-parametric algorithm work that can be used for classification and regression task. It prdicts the label of a query point by looking at the k closet training points, usually measured using Euclidean distance, and taking a majority vote or average. sonce KNN does'nt learn an explicit model during training and jsut stores the data, it's often referred to as a lazy learner. Choosing a very small value of k typically makes the model more robust to noisy data. Geature scaling matters a lot for KNN since it relies on distance calcuations. KNN has a training time compexity of 0(n log n) due to the tree structures used internally."""

retrieval_context = [
    "K-Nearest Neighbors is a non-parametric, instance-based learning algorithm used for both classification and regression.",
    "It works by finding the k closest training examples to a query point, typically using Euclidean distance, and predicting based on majority vote (for classification) or averaging (for regression).",
    "The algorithm does not build an explicit model during training - it simply sotres the trainng data, which is why it is called a 'lazy learner'.",
    "The choice of k significantly affects performace: a small k makes the model sensitive to noise, while large k can smooth out decision boundaries too much.",
    "Feature scaling is important for KNN because distance calculations are sensitive to the magnitude of feature values."
]

# test case
test_case = LLMTestCase(
    input = rag_input,
    retrieval_context= retrieval_context,
    actual_output=actual_output
)

# metric
context_precision = FaithfulnessMetric(verbose_mode=True,
penalize_ambiguous_claims=True,
truths_extraction_limit=7)

# test the metric
context_precision.measure(test_case = test_case)