text_file = './Texts'

fr = open(text_file + '/text1.txt', "r", encoding='utf-8')
text = fr.read()

education = text.split("EDUCATION")
work_experience = education[1].split('WORK EXPERIENCE')

print(education[1])
print('---------------------------')
print(work_experience[1])
