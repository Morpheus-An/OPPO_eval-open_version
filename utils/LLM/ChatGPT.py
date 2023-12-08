import sys
import os
if sys.platform == 'win32':
    path_sep = '\\'
else:
    path_sep = '/'
import openai


# openai.api_base = 'https://api.closeai-asia.com/v1'
with open('./ChatGPT_API_Key.txt', 'r') as f:
    openai.api_key = f.read().strip()
print(openai.api_key)

DEFAULT_SYSMESS = ""
GPT3_5 = "gpt-3.5-turbo"
GPT4 = "gpt-4"
GPT3_5_INST = "gpt-3.5-turbo-instruct"
DEBUG = False

DefaultSystemMessage = ''


def _call_chat_api(messages, model, temperature, top_p):
    response = openai.ChatCompletion.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    return response


def chat(utters, sys_mess=DefaultSystemMessage, top_p=None, temp=1.0, model=GPT3_5, debug=DEBUG):
    if type(utters) == str:
        utters = [utters]
    messages = [
        {'role': 'system', 'content': sys_mess},
    ]
    # make sure the last utterance is from the user
    if len(utters) % 2 == 1:
        roles = ['user', 'assistant'] * len(utters)
    else:
        roles = ['assistant', 'user'] * len(utters)
    for role, utter in zip(roles, utters):
        messages.append({'role': role, 'content': utter})

    if debug:
        return _call_chat_api(messages, model, temp, top_p)
    else:
        response = "NO response received"
        max_try = 3
        for i in range(max_try):
            response = _call_chat_api(messages, model, temp, top_p)
            reason = response['choices'][0]['finish_reason']
            if reason in ['stop']:
                return response['choices'][0]['message']['content']
            elif reason in ['length']:
                raise Exception(
                    f"openai {model} stop generation due to length constrain.\n" +
                    f"The last response is \n{response}")
            else:
                print("unexpected response from openai .\n\n Try again ...")
    # all attempts failed
    raise Exception(f'Something Wrong when calling openai {model} API' +
                    f"The last response is \n{response}")


def test_chat(query, sys_mess=DEFAULT_SYSMESS, top_p=None, temp=1.0, model=GPT3_5):
    response = openai.ChatCompletion.create(
        messages=[
            {"role": "system", "content": sys_mess},
            {"role": "user", "content": query},
        ],
        model=model,
        temperature=temp,
        top_p=top_p,
    )
    return response


def complete(query, top_p=None, temp=1.0,  max_tokens=3072, model=GPT3_5_INST):
    response = "NO response received"
    max_try = 3
    for i in range(max_try):
        response = openai.Completion.create(
            prompt=query,
            model=model,
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens
        )
        reason = response['choices'][0]['finish_reason']
        if reason in ['stop']:
            return response['choices'][0]['text']
        elif reason in ['length']:
            raise Exception(
                f"openai {model} stop generation due to length constrain.\n" +
                f"The last response is \n{response}")
        else:
            print(
                f"unexpected response from openai {response}.\n\n Try again ...")
    # all attempts failed
    raise Exception(f'Something Wrong when calling openai {model} API' +
                    f"The last response is \n{response}")


def test_complete(query, top_p=None, temp=1.0,  max_tokens=128, model=GPT3_5_INST):
    response = openai.Completion.create(
        prompt=query,
        model=model,
        temperature=temp,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return response


if __name__ == '__main__':
    print(chat('hello'))
