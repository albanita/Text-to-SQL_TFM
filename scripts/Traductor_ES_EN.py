import requests
import os


class Traductor_ES_EN():

    def __init__(self):
        token = os.getenv("HuggingFace_Token")
        self.API_URL = "https://api-inference.huggingface.co/models/Berly00/mbart-large-50-spanish-to-english"
        self.headers = {"Authorization": f"Bearer {token}"}

    def __query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()

    def translate(self, text):
        try:
            output = self.__query({
                "inputs": text,
            })

            # print("text:", text)
            # print("output:", output)

            translated_text = output[0]['generated_text']
            return translated_text
        except Exception as e:
            raise f"Error al traducir. Por favor, intente de nuevo.\n {str(e)}"
