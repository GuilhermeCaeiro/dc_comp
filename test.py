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
                ]

training_patterns = [row[0] for row in training_set]
expected_outputs = [row[1] for row in training_set]

wann = Wisard(2, 3546)
wann.train(training_patterns, expected_outputs)


print("\n\n\n")

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
for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")
for pattern in test_set:
    print("Evaluating pattern ", pattern)
    print("Result: ", wann.predict(pattern[0]), "Expected: ", pattern[1], "\n")


