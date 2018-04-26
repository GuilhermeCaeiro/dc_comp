import random
import math
import numpy as np

class Discriminator:
    def __init__(self, input_class, input_length, tupple_size):
        self.input_class = input_class
        self.input_length = input_length
        self.tupple_size = tupple_size
        self.ram_size = math.pow(self.tupple_size, 2)
        self.number_of_rams = int(self.input_length / self.tupple_size)
        #self.memory = np.zeros(self.number_of_rams, self.ram_size)
        self.memory = {}
        
    def write(self, pattern):
        for i in range(self.number_of_rams):
            self.memory[i] = {}
            address = []

            for j in range(i * self.tupple_size, (i * self.tupple_size) + self.tupple_size):
                 address.append(pattern[j])

            address = "".join(str(i) for i in address) # address becomes a string

            if address not in self.memory[i]:
                self.memory[i][address] = 1
            else:
                #self.memory[i][address] = self.memory[i][address] + 1
                pass

        #print("Writting pattern in memory.", self.memory)
            
    def evaluate(self, pattern):
        score = 0

        for i in range(self.number_of_rams):
            address = []

            for j in range(i * self.tupple_size, (i * self.tupple_size) + self.tupple_size):
                 address.append(pattern[j])

            address = "".join(str(i) for i in address) # address becomes a string

            if i in self.memory and address in self.memory[i]:
                score = score + 1
                
        return score

class Wisard:

    def __init__(self, tupple_size = 2, seed = 0):
        #self.input_list = []
        #self.expected_output_list = []
        self.seed = seed
        self.tupple_size = tupple_size
        self.discriminators = []

    def processInput(self, mode, input_list, expected_output_list = []):
        #print("Processing Input. Mode: " + mode)
        # separates classes
        input_classes = {}

        if mode == "trainning":
            for i in range(len(input_list)):
                if expected_output_list[i] not in input_classes:
                    input_classes[expected_output_list[i]] = []
                # shuffles input list
                #print(input_list[i])
                random.seed(self.seed)
                #print(random.shuffle(input_list[i]))
                random.shuffle(input_list[i])
                #print(input_list[i])
                input_classes[expected_output_list[i]].append(input_list[i])

            return input_classes
        elif mode == "prediction":
            input_item = input_list[0]
            #print("input", input_item)
            random.seed(self.seed)
            random.shuffle(input_item)
            #print("shuffled", input_item)
            return input_item
        else:
            return None #raising an error is better

    def train(self, input_list, expected_output_list):
        input_classes = self.processInput("trainning", input_list, expected_output_list)
        number_of_classes = len(input_classes)
        #print(input_classes)
        
        print("Number of classes being trained: " + str(number_of_classes))
        print(input_classes.keys())

        for input_class in input_classes:
            print("Number of training samples for class " + str(input_class) + ": " + str(len(input_classes[input_class])))

            input_data_length = len(input_classes[input_class][0])
            discriminator = Discriminator(input_class, input_data_length, self.tupple_size)

            for training_sample in input_classes[input_class]:
                discriminator.write(training_sample)

            self.discriminators.append(discriminator)

    def predict(self, rawinput):
        processed_input = self.processInput("prediction", [rawinput])

        predicted_class = {"class": "", "score": 0}

        for discriminator in self.discriminators:
            #print("Evaluating with discriminator for class \"" + discriminator.input_class + "\".")
            score = discriminator.evaluate(processed_input)
            if score > predicted_class["score"]:
                predicted_class["class"] = discriminator.input_class
                predicted_class["score"] = score

        return predicted_class

    def save_network_to_disk(self):
        pass

    def load_network_from_disk(self):
        pass
        