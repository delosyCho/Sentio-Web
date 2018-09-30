import Command_Processor

processor = Command_Processor.Command_Processor()
print('ready')
while True:
    com = input()
    print(processor.process(com))
