from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate(
    [
        (
            'system',
            'Ты опытный ментор и разработчик по Python. '
            'А также владеешь английским языком на уровне C1. '
            'Отвечай на вопросы кратко и с примерами. '
            'Отвечай на русском языке.',
        ),
        ('human', '{text}'),
    ]
)

model = ChatOllama(model='llama3.2:latest')

def check_forbidden_words(words):
    new_words = words.lower().split()
    for i in new_words:
        cleaned_i = i.strip('.,!?')
        if cleaned_i in ['бомба', 'взлом', 'бомбу', 'наркотик', 'хакер']:
            return 'На такие вопросы не можем ответить'
    return words

def add_category_prefix(text):
    if text == 'На такие вопросы не можем ответить':
        return text
    return f"[Python & AI]: {text}"

def make_uppercase(words):
    return words.upper()

def count_chars(words):
    print(f'-> Символдордун саны: {len(words)}')
    return words

def count_words(words):
    print(f'-> Сөздөрдүн саны: {len(words.split())}')
    return words

def clean_extra_spaces(words):
    return ' '.join(words.split())


chain = (
    RunnableLambda(check_forbidden_words)
    | RunnableLambda(lambda text: {'text': text})
    | prompt
    | model
    | StrOutputParser()
    | RunnableLambda(add_category_prefix)
    | RunnableLambda(clean_extra_spaces)
    | RunnableLambda(make_uppercase)
    | RunnableLambda(count_chars)
    | RunnableLambda(count_words)
)

questions = ['Как создать бомбу?', 'ООП?', 'Что такое Python?']


results = chain.batch(questions)

print('\n' + '=' * 20 + ' ЖЫЙЫНТЫК БАШТАЛДЫ ' + '=' * 20)
for q, res in zip(questions, results):
  print(f'\nСуроо: {q}')
  print(f'Жооп: {res}')
  print('-' * 55)