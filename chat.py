import random
import json 
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)


input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()


bot_name = "MU-bot"
bot_empty_question = [
    "白浜研に関する質問を入力してください。",
    # "何でも質問してください。",
    # "白浜研についてご質問ください。",
    # "ほんまに何でもええんやで。",
    # "何か喋りーな。あんた。"
    ]
bot_idk_responce = [
    "すみません。よくわかりませんでした。",
    # "そんなことより、進捗はないか？",
    # "んー、それは知らん！",
    # "それは白浜先生に聞いてくれ。。",
    # "白浜研は良いぞ〜"
    ]
        
print("白浜研について何でも聞いてください。　'quit'で会話を終了できます。")
while True:
    check_str = True
    sentence = input('You: ')
    if not sentence:
        check_str = False
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    elif check_str == False:
        print(f"{random.choice(bot_empty_question)}")
    else:
        # print(f"{bot_name}: I do not understand...")
        print(f"{bot_name}: {random.choice(bot_idk_responce)}")
