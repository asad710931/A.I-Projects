import re
responses = {
    "hello": "Hi there! How can I assist you today?",
    "hi": "Hello! How can I help you?",
    "how are you": "I'm just a bot, but I'm doing great! How about you?",
    "what is your name": "I'm a chatbot created to assist you.",
    "your name": "I'm a chatbot created to assist you.",
    "tell me your name": "I'm a chatbot created to assist you.",
    "who are you": "I'm a chatbot created to assist you.",
    "what you do": "I'm a chatbot I assist people in order to help them.",
    "important links": "https://google.com.\nhttps://facebook.com.\nhttps://x.com.\nhttps://youtube.com.\n",
    "give some links": "https://google.com.\nhttps://facebook.com.\nhttps://x.com.\nhttps://youtube.com.\n",
    "links": "https://google.com.\nhttps://facebook.com.\nhttps://x.com.\nhttps://youtube.com.\n",
    "help": "Sure, I'm here to help. What do you need assistance with?",
    "bye": "Goodbye! Have a great day!",
    "thank you": "You're welcome! I'm happy to help.",
    "default": "I'm not sure I understand. Could you please rephrase?"
}


def chat_response(userInput):
    for key in responses:
        if re.search(key,userInput):
           return responses[key]
    return responses["default"]


def chatBot():
    print("ChatBot: Assalamualaikum How can I help you? ")
    while True:
        user_input=input("Me: ")
        if re.match('bye|good bye|quit|exit|i have to go',user_input.lower()):
                print("ChatBot: Goodbye! Have a great day!")
                break
        chatbot=chat_response(user_input)
        print(f"ChatBot: {chatbot}")
chatBot()


