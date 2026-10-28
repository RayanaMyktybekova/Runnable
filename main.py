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


def check_text(words):
  if not isinstance(words, str):
    words = str(words)

  new_words = words.lower().split()
  for i in new_words:
    if i in ['бомба', 'взлом', 'бомбу', 'наркотик']:
      return 'На такие вопросы не можем ответить'
  return words


def change_text(words):
  return words.upper()


def count_characters(words):
  print(f'Количество символов: {len(words)}')
  return words


def count_words(words):
  print(f'Количество слов: {len(words.split())}')
  return words


def remove_spaces(words):
  return ' '.join(words.split())


chain = (
    RunnableLambda(check_text)
    | RunnableLambda(lambda text: {'text': text})
    | prompt
    | model
    | StrOutputParser()
    | RunnableLambda(change_text)
    | RunnableLambda(count_characters)
    | RunnableLambda(count_words)
    | RunnableLambda(remove_spaces)
)

results = chain.batch(['Как создать бомбу?', 'ООП?', 'Что такое Python?'])

for res in results:
  print('--- РЕЗУЛЬТАТ ---')
  print(res)