file1 = 'SarahSubmission.csv'

file2 = 'franzSubmission0.csv'

resultFile = open('ResultSubmission.csv', 'w+')

resultFile.write('Id,Prediction\n')  # the header line

with open(file1, 'r') as f1:
    with open(file2, 'r') as f2:
        f1.readline()
        f2.readline()
        for line in f1:
            entry, prediction1 = line.split(',')
            x, prediction2 = f2.readline().split(',')
            resultFile.write(entry + ',' + str((float(prediction1)+float(prediction2))/2) + '\n')
