SMSCollection = open('SMSSpamCollection.txt')

spamExclamationCounter = 0

SMSDictionary = {}
SMSDictionary['spam'] = [0, 0]
SMSDictionary['ham'] = [0, 0]

for SMS in SMSCollection:
    SMS = SMS.rstrip()
    words = SMS.split()
    if(words[0]=='spam'):
        counter = SMSDictionary['spam']
        counter[0] += (len(words) - 1)
        counter[1] += 1
        SMSDictionary['spam']
        if(words[-1].endswith('!')):
            spamExclamationCounter += 1
    else:
        counter = SMSDictionary['ham']
        counter[0] += (len(words) - 1)
        counter[1] += 1
        SMSDictionary['ham']

print('Average words in spam SMS: ', SMSDictionary['spam'][0]/SMSDictionary['spam'][1])
print('Average words in ham SMS: ', SMSDictionary['ham'][0]/SMSDictionary['ham'][1])
print('Spam exclamation mark count: ', spamExclamationCounter)
