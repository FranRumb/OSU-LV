wordDictionary = {}

songLyrics = open('song.txt')

for line in songLyrics:
    line = line.rstrip()
    line = line.replace(',', '')
    line = line.lower()
    words = line.split()
    for word in words:
        if(word in wordDictionary):
            number = wordDictionary[word]
            number = number + 1
            wordDictionary[word] = number
        else:
            wordDictionary[word] = 1

print(wordDictionary)