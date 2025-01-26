from includes.model_functions import all_nested_dichotomies

elements = (1, 2, 3)
dichotomies = list(all_nested_dichotomies(elements))
for i, nd in enumerate(dichotomies, 1):
    print(f"Dichotomy {i}: {nd}")