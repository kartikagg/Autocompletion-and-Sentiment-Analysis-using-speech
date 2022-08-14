from sklearn import pipeline
import speech_recognition
import pyttsx3
import pickle
from tensorflow.keras.models import load_model
import numpy as np


pipeline = pickle.load(open('trained_model.sav','rb'))
model = load_model('next_words.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))
recognizer = speech_recognition.Recognizer()



def Predict_Next_Words(model, tokenizer, text):



    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""
    
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
  
    print(predicted_word)
    return predicted_word

while True:
    try:

        with speech_recognition.Microphone() as mic:

            recognizer.adjust_for_ambient_noise(mic, duration=0.4)
            audio = recognizer.listen(mic)

            text = recognizer.recognize_google(audio, language = 'en-IN', show_all = True)
            print(text)
            #text = text.lower()
            text_auto =  text['alternative'][0]['transcript']

            print(f"Recognized {text_auto}")

            choice = int(input('Enter 1 for autocomplete or 0 for sentiment'))

            

            if choice == 0:

                l = [text['alternative'][0]['transcript']]
                b = pipeline.predict(l)
                if b[0]==0:
                    print("The user gave a negative statement")
                else:
                    print("The user gave a positive statement")

                print(f"Confidence {text['alternative'][0]['confidence']}")
                break
            elif choice == 1:

                while(True):

                
  
                    if text_auto == "zero":
                        print("Execution completed.....")
                        break
  
                    else:
                        try:
                            text_auto = text_auto.split(" ")
                            text_auto = text_auto[-3:]
                            print(text_auto)
        
                            Predict_Next_Words(model, tokenizer, text_auto)
                            break
          
                        except Exception as e:
                            print("Error occurred: ",e)
                            continue 

            

        
    except speech_recognition.UnknownValueError() as mic :

        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
        continue
print("Thank you!")

