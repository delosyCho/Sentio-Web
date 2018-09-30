import Wiki_Dataholder

dataset = Wiki_Dataholder.Data_holder()

while True:
    try:
        print('입력하시오')
        idx = int(input())

        print('ans start', dataset.Labels_start[idx])
        print('ans stop', dataset.Labels_stop[idx])

        print(dataset.Questions[idx])
        print(dataset.Paragraphs[idx])

        print(dataset.Start_Label_Input[idx])

    except:
        0