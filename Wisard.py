import random
import math
import numpy as np
import copy

class Discriminator:
    def __init__(self, input_class, input_length, tupple_size, bleaching = False):
        self.input_class = input_class
        self.input_length = input_length
        self.tupple_size = tupple_size
        self.ram_size = math.pow(self.tupple_size, 2)
        self.number_of_rams = int(self.input_length / self.tupple_size)
        self.bleaching = bleaching
        #self.memory = np.zeros(self.number_of_rams, self.ram_size)
        self.memory = {}
        
    def write(self, pattern):
        for i in range(self.number_of_rams):
            if i not in self.memory:
                self.memory[i] = {}
            address = []

            for j in range(i * self.tupple_size, (i * self.tupple_size) + self.tupple_size):
                 address.append(pattern[j])

            address = "".join(str(i) for i in address) # address becomes a string

            if address not in self.memory[i]:
                #print("Memory Created")
                self.memory[i][address] = 1
            elif address in self.memory[i] and self.bleaching:
                #print("Memory Incremented")
                self.memory[i][address] = self.memory[i][address] + 1
                #pass

        #print("Writting pattern in memory.", self.memory)
            
    def evaluate(self, pattern, bleaching_threshold = 0):
        score = 0

        for i in range(self.number_of_rams):
            address = []

            for j in range(i * self.tupple_size, (i * self.tupple_size) + self.tupple_size):
                 address.append(pattern[j])

            address = "".join(str(i) for i in address) # address becomes a string

            if i in self.memory and address in self.memory[i]:
                if self.bleaching and self.memory[i][address] >= bleaching_threshold:
                    print("Threshold: ", bleaching_threshold, "Somado: ", 1)
                    score = score + 1
                elif self.bleaching and self.memory[i][address] < bleaching_threshold:
                    print("Threshold: ", bleaching_threshold, "Somado: ", 0)
                    continue
                else:
                    score = score + 1
                
        return score 

class Wisard:

    def __init__(self, tupple_size = 2, seed = 0, bleaching = False):
        #self.input_list = []
        #self.expected_output_list = []
        self.seed = seed
        self.bleaching = bleaching
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
                input_item = copy.deepcopy(input_list[i])
                #print("Original pattern: ", input_item)
                random.seed(self.seed)
                random.shuffle(input_item)
                #print("Shuffled pattern: ", input_item)
                input_classes[expected_output_list[i]].append(input_item)

            return input_classes
        elif mode == "prediction":
            #input_item = input_list[0]
            input_item = copy.deepcopy(input_list[0])
            #print("Original pattern: ", input_item)
            random.seed(self.seed)
            random.shuffle(input_item)
            #print("Suffled pattern: ", input_item)
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
            discriminator = Discriminator(input_class, input_data_length, self.tupple_size, self.bleaching)

            for training_sample in input_classes[input_class]:
                discriminator.write(training_sample)

            self.discriminators.append(discriminator)

    def predict(self, rawinput):
        processed_input = self.processInput("prediction", [rawinput])

        result_achieved = False

        predicted_classes = []
        current_threshold = 0

        discriminators_to_evaluate = self.discriminators

        while not result_achieved:
            predicted_classes = [{"discriminator": None, "score": 0}]
            #predicted_class = {"class": "", "score": 0}

            for discriminator in discriminators_to_evaluate:
                print("Evaluating with discriminator for class \"" + discriminator.input_class + "\".")
                score = discriminator.evaluate(processed_input, current_threshold)
                print("Score for discriminator", discriminator.input_class, ": ", score)
                if score > predicted_classes[0]["score"]:
                    predicted_classes = [{"discriminator": discriminator, "score": score}]
                elif score == predicted_classes[0]["score"]:
                    predicted_classes.append({"discriminator": discriminator, "score": score})


            if not self.bleaching:
                print("Nada de Bleaching, sefinir!")
                result_achieved = True
            elif self.bleaching and len(predicted_classes) > 1:
                print("Yahho!", len(predicted_classes))
                if predicted_classes[0]["score"] == 1:
                    print("Score 1")
                    result_achieved = True


                current_threshold = current_threshold + 1

                discriminators_to_evaluate = []
                for predicted_class in predicted_classes:
                    discriminators_to_evaluate.append(predicted_class["discriminator"])

            elif self.bleaching and len(predicted_classes) == 1:
                print("Pimba!")
                result_achieved = True
            else:
                print("Error predicting class.")
                break

        # If the method ends with more than one class as possible, it just returns the first one
        return {"class": predicted_classes[0]["discriminator"].input_class, "score": predicted_classes[0]["score"]}

    def deactivate_bleaching(self):
        self.bleaching = False

    def show_mental_map(self):
        pass

    def save_network_to_disk(self):
        pass

    def load_network_from_disk(self):
        pass
        