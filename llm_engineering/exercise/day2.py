from OpenChat import OpenChat


if __name__ == '__main__':
    # model = 'deepseek-r1:1.5b' # 'llama3.2'
    # openChat: OpenChat = OpenChat(model=model, call_type='openai')
    openChat: OpenChat = OpenChat(call_type='openai') # openai, ollama

    # message = "Describe some of the business applications of Generative AI"
    message = "Please give definitions of some core concepts behind LLMs: a neural network, attention and the transformer"
    response = openChat.send(message=message, stream=False)
    print(response.message.content)

