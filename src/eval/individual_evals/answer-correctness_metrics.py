from langsmith.schemas import RubricItemConditionRule
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case.llm_test_case import SingleTurnParams
from dotenv import load_dotenv

# load the api key
load_dotenv()

# define the custom metic

answer_correctness = GEval(
    name = "answer correctness",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    criteria="""Evaluate the LLM response based on correctness of answer. Compare the 'expected_output' with the 'actual_output'. Penalize wrong facts strictly""",
    rubric=[Rubric(score_range=(0,5),  expected_outcome="Answer has incorrect facts"),
            Rubric(score_range=(6,9), expected_outcome="Answer is mostly correct but has minor differences"),
            Rubric(score_range=(10,10), expected_outcome=r"100% correct")],
            verbose_mode=True
)

test_case = LLMTestCase(
    input="What specific complex tasks can Siri perform using its on-screen awareness and multi-app functionalities?",
    actual_output="Siri can perform tasks such as understanding the content displayed on your screen and providing relevant information, like finding the meaning of a word in an article you are reading. Additionally, Siri can design specific items like a wedding outfit if you have given it access to your shopping apps. It can also help in planning your day by knowing your dining habits and preferences. Moreover, Siri can retrieve specific photos from your library based on criteria you set, such as photos clicked last week.",
    expected_output="Siri can perform several complex tasks using its on-screen awareness and multi-app functionalities. For example, if you're watching a video and pause it, you can ask Siri to find a specific item, like a jacket worn by a character, and it will identify the item based on the screen content. Additionally, if you're reading an article and encounter an unfamiliar word, you can request Siri to find its meaning in the context of that article. Siri can also execute multi-app actions, such as forwarding an email to your boss or scheduling a meeting in your calendar, provided you grant the necessary app permissions."
)

print(answer_correctness.measure(test_case= test_case))