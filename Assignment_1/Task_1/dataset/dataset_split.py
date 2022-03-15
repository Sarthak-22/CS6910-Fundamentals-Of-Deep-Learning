lines = []
import random
with open("func_app1.csv","r") as f:
    lines = f.readlines()
random.shuffle(lines[1:])
train_lines = lines[1:351]
valid_lines = lines[351:401]
test_lines = lines[401:501]

with open("train.csv","w") as f:
    f.write(lines[0])
    for line in train_lines:
        f.write(line)

with open("validation.csv","w") as f:
    f.write(lines[0])
    for line in valid_lines:
        f.write(line)

with open("test.csv","w") as f:
    f.write(lines[0])
    for line in test_lines:
        f.write(line)



