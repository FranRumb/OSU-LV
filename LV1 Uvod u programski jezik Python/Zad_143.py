from statistics import mean

numberList = []

while True:

    try:
        inputtedValue = input()
        if(inputtedValue == 'Done'):
            break
        numberList.append(float(inputtedValue))
    except:
        print('Please enter a number or Done')

print('Number of values: ', len(numberList))
print('Mean value: ', mean(numberList))
print('Minimal value: ', min(numberList))
print('Maximal value: ', max(numberList))
numberList.sort()
print(numberList)

