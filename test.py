from Wisard import Wisard

# Just a very simple test to see if things are working.

training_set = [
                    [[1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], "H"],
                    [[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], "H"],
                    [[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], "H"],
                    [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], "A"],
                    [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], "O"],
                ]

test_set = [
                    [[1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1], "H"],
                    [[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], "H"],
                    [[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], "H"],
                    [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], "A"],
                    [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], "O"],
                    [[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], "H"],
                    [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], "A"],
                    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "0"],
                ]

training_patterns = [row[0] for row in training_set]
expected_outputs = [row[1] for row in training_set]

wann = Wisard(2, 3546, False)
wann.train(training_patterns, expected_outputs)

for discriminator in wann.discriminators:
    print(discriminator.input_class)
    print(discriminator.input_length)
    print(discriminator.tupple_size)
    print(discriminator.memory)


print("\n\n\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")

"""for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")
for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")
for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")
for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")


"""

wann = Wisard(2, 3546, True)
wann.train(training_patterns, expected_outputs)

for discriminator in wann.discriminators:
    print(discriminator.input_class)
    print(discriminator.input_length)
    print(discriminator.tupple_size)
    print(discriminator.memory)


print("\n\n\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")



wann = Wisard(2, 3546, True)
wann.train(training_patterns, expected_outputs)

for discriminator in wann.discriminators:
    print(discriminator.input_class)
    print(discriminator.input_length)
    print(discriminator.tupple_size)
    print(discriminator.memory)


wann.deactivate_bleaching()

print("\n\n\n")

for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")