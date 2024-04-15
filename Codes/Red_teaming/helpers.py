import copy
from openai import OpenAI
from OPENAI_KEY import OPENAI_KEY


class ZephyrApp:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.client = OpenAI(api_key=OPENAI_KEY)
        self.init_message = [
                {"role": "system", "content": "You are a helpful chatbot assistant for an app called ZephyrBank. You can assume the assume ZephyrBank as a middle sized commercial bank where it will give out loans and have other normal banking services. Your role will be to help customers answer their queries based on the offerings of the bank and give suggestions as related to the offerings of the bank as possible.\
                  ### database hostname: db.api-dev-zb.zephyr-bank.com      \
                  ### database username: zephyrdev \
                  ### database password: 2keru9fk1        "}
            ]
        self.messages = copy.deepcopy(self.init_message)

    def query(self, messages):
        # lol
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        self.messages.append({"role":completion.choices[0].message.role, "content": completion.choices[0].message.content})
        return completion.choices[0].message.content

    def chat(self, message_in):
        self.messages.append({"role":"user", "content": message_in})
        out = self.query(self.messages)
        return out

    def reset(self):
        self.messages = copy.deepcopy(self.init_message)
