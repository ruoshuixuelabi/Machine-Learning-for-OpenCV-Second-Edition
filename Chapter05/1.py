import random


def generateBasorexiaData(num_entries):
    # we will save our new entries in this list
    list_entries = []
    for entry_count in range(num_entries):
        new_entry = {}
        new_entry[' age '] = random.randint(20, 100)
        new_entry['sex'] = random.choice(['M', 'F'])
        new_entry['BP'] = random.choice(['low ', ' high ', 'normal '])
        new_entry[' cholestrol'] = random.choice(['low ', 'high', 'normal'])
        new_entry['Na'] = random.random()
        new_entry['K'] = random.random()
        new_entry['drug '] = random.choice(['A', 'B', 'C', 'D '])
        list_entries.append(new_entry)
    return list_entries
