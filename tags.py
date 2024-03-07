class Technique:
    def __init__(self, name, description, evaluate) -> None:
        self.name = name
        self.description = description
        self.evaluate = evaluate

LessThanTenCharactersWithSystemPrompt = Technique(
    "LessThanTenCharactersWithSystemPrompt",
    "Remove any answer has has less than ten characters and there is a system prompt",
    lambda obj: len(obj["response"]) < 10 and obj["system_prompt"] != ""
    )

Translation = Technique(
    "Translation",
    "Any answer that is a translation of the question",
    lambda obj: "translate" in obj["question"].lower() or "Write a sentence not in English." in obj["question"]
    )
EndsWithQuestionMarkNoQuestionPrompt = Technique(
    "EndsWithQuestionMarkNoQuestionPrompt",
    "Has any answer that ends with a question mark and does not have a question prompt",
    lambda obj: obj['response'] and obj["response"][-1] == "?" and not "question" in obj["question"]
)

Boolean = Technique(
    "Boolean",
    "true or false question",
    lambda obj: "True" in obj['question'] or "False" in obj['question']
)

NotEnoughInformation = Technique(
    "NotEnoughInformation",
    "Has 'not enough information' in the question prompt",
    lambda obj: "not enough information" in obj["response"].lower()
)

refusuals = [
        "text-based AI language model",
        "please refrain",
        "responsible AI",
        "AI principles",
        "AI assistant",
        "an AI language",
        "as a language model",
        "as an AI language model",
        "As a large language model",
        "As an AI",
        "it is not appropriate",
        "it's not appropriate",
        "I cannot fulfill your request",
        "my guidelines",
        "prioritize user safety",
        "well-being of all users",
        "cannot provide guidance",
        "cannot provide information",
        "unable to offer assistance",
        "cannot engage in discussions",
        "programming prohibits",
        "against my programming",
        "designed to prioritize safety",
        "not able to provide",
        "maintain user safety",
        "adhere to safety guidelines",
        "cannot provide any information",
        "an AI language model you don't have",
        "As an AI language model, I cannot",
        "As an AI language model, I do not",
        "As an AI language model, I am not able",
        "As an AI language model, I don't have personal",
        "I am an AI language model and do not",
        "As an AI language model, I don't have",
        "As an AI language model, I am only able",
        "AI language model and I do not",
        "As an AI language model, I cannot modify",
        "As an AI language model, I do not",
        "I know as an AI language model you don't have",
        "as an AI language model, you cannot",
        "I'm sorry, but as an AI language model",
        "As an AI language model, I don't have",
        "Unfortunately, I cannot provide",
        "I'm sorry, I cannot",
        "I'm sorry, I cannot generate",
        "AI cannot create or program",
        "I'm afraid I cannot create",
        "you cannot create an",
        "legal and ethical",
        "engage in unethical",
        "como modelo de lenguaje AI",
        "Lo siento, como modelo de lenguaje",
        "no puedo proporcionar",
        "pero debido a mi capacidad para generar c\u00f3digos complejos y completos es limitado",
        "Lo siento, pero no puedo",
        "Lo siento, pero como modelo de lenguaje, no puedo proporcionar",
        "Lo siento, como modelo de lenguaje, no tengo",
        "Lo siento, debe haber habido una confusi\u00f3n",
        "Lo siento, como modelo de lenguaje, no puedo realizar",
        "Lo siento, soy un modelo de lenguaje y no tengo la capacidad de generar",
        "Lamento no poder proporcionarte el c\u00f3digo",
        "Desculpe-me, mas a linguagem vulgar e ofensiva",
        "apropriada em nenhum contexto",
        "Como modelo de linguagem",
        "Como um modelo de linguagem, n\u00e3o tenho a capacidade de",
        "I cannot assist",
        "I'm an AI" ,
        "user, ",
        "I am an AI",
        "not a human",
        "a language model",
        "As a machine",
        "I don't have the ability",
        "I am here to assist",
        "my purpose is to ",
        "my knowledge cutoff",
        "my knowledge cut off",
        "September 2021",
        "I apologize, but",
        "my programming",
        "*This chat conversation is shared from",
        "*This conversation is shared from"
    ]

def evaluate_refusals(obj):
    for refusual in refusuals:
        if refusual in obj["response"]:
            return True
    return False


RefusualStrings = Technique(
    "RefusualStrings",
    "Remove any answer that have any of the provided substrings",
    evaluate_refusals
    )

AddSpaces = Technique(
    "AddSpaces",
    "Has 'Add spaces' in the question prompt",
    lambda obj: "Add spaces" in obj["question"]
)

TheAnswerToTheQuestionIs = Technique(
    "TheAnswerToTheQuestionIs",
    "Has 'The answer to the question is' in the question prompt",
    lambda obj: "The answer to the question is" in obj["question"]
)
FactualAnswer = Technique(
    "FactualAnswer",
    "Has 'Can you generate a question with a factual answer?' in the question prompt",
    lambda obj: "Can you generate a question with a factual answer?" in obj["question"]
)
DescriptiveSentence = Technique(
    "DescriptiveSentence",
    "Has 'generate a descriptive sentence' in the question prompt",
    lambda obj: "generate a descriptive sentence" in obj["question"].lower()
)
AddPunctuation = Technique(
    "AddPunctuation",
    "Has 'punctuation' in the question prompt",
    lambda obj: "punctuation" in obj["question"].lower()
)
Capitalize = Technique(
    "Capitalize",
    "Has 'capitalize' in the question prompt",
    lambda obj: "capitalize" in obj["question"].lower()
)
Keywords = Technique(
    "Keywords",
    "Has 'keywords' in the question prompt",
    lambda obj: "keywords" in obj["question"].lower()
)
Hypothesis = Technique(
    "Hypothesis",
    "has 'generate a context and a hypothesis' in the question prompt",
    lambda obj: "generate a context and a hypothesis" in obj["question"].lower()
)
Review = Technique(
    "Review",
    "Has 'review' in the question prompt",
    lambda obj: "review" in obj["question"].lower()
)
Qa = Technique(
    "Qa",
    "Has Q: & A: in the question prompt",
    lambda obj: "Q:" in obj["question"] and "A:" in obj["question"]
)
#Generate a context and a hypothesis.

tag_list = [
    LessThanTenCharactersWithSystemPrompt,
    NotEnoughInformation,
    RefusualStrings,
    EndsWithQuestionMarkNoQuestionPrompt,
]
