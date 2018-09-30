import dDANN
import Wiki_Dataholder
import QA_Model

import tensorflow as tf

excute = 2

classifier = dDANN.One_Model()

if excute == 0:
    classifier.training_prediction_index(700000, False, Model=0, is_DANN=False, source=True, target=False)
elif excute == 1:
    classifier.test_target(Model=0, display=False)
elif excute == 2:
    classifier.test_source(Model=2, display=True)
elif excute == 3:
    classifier.check(is_BIDAF=True)
elif excute == 4:
    classifier.check2(is_BIDAF=True)
else:
    dataset = Wiki_Dataholder.Data_holder()
    for i in range(1000):
        print(dataset.exo_Questions[i])
        print(dataset.Exo_Questions_Input[i])

        print()
        print(dataset.exo_Paragraphs[i])
        print(dataset.Exo_Start_Label_Input[i])
        print(dataset.Exo_Stop_Label_Input[i])
