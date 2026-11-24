from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from dotenv import load_dotenv


load_dotenv()

rag_input = "How does bagging work?"
expected_ouput = """Bagging, short for bootsrap aggregating, works by creating multiple     subsets of the training data through random sampling with replacement. A separate model is trained on each subset independently. For classification, the final prediction is made by majority voting across all models. For regression, the predictions are averaged. This process reduces variance and helps prevent overfitting. """
retrieval_context = [
    "Bagging creates multiple bootstrap samples from the training dataset - each sample is drawn randomly with replacement, meaning some data points may appear multiple times while others may be left out.",
    "In bagging, an independent model is trained on each bootstrap sample. For regression tasks, the final output is the average of all individual model predictions."
]

# test case
test_case = LLMTestCase(
    input = rag_input,
    retrieval_context=retrieval_context,
    expected_output=expected_ouput,
    actual_output=""
)

# metric
recall_mertic = ContextualRecallMetric(verbose_mode=True)

# test the metric
recall_mertic.measure(test_case = test_case)