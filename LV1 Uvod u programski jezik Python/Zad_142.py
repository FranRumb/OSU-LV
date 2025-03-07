grade = 0.0



while 1:
    grade = input()
    try:
        grade = float(grade)

        if(grade > 1.0 or grade < 0.0):
            raise ValueError('Number is too big')

        if(grade >= 0.9):
            print('A')
        elif(grade >= 0.8):
            print('B')
        elif(grade >= 0.7):
            print('C')
        elif(grade >= 0.6):
            print('D')
        elif(grade < 0.6):
            print('F')

    except ValueError as e:
        print('Enter only numbers in range from 0.0 to 1.0')
    else:
        break


