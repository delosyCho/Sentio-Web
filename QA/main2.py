import Sentence_Data_Processor
import Sentence_Representation_Model

statement = 3

if statement == -7:
    model = Sentence_Representation_Model.DMA_NET(isEvaluate=True)
    model.get_test_data_result_()
    input()
if statement == -6:
    model = Sentence_Representation_Model.DMA_NET(isEvaluate=True)
    model.training_prediction_index(training_epoch=90000, is_continue=True)
    input()
if statement == -5:
    model = Sentence_Representation_Model.DMA_NET(isEvaluate=True)
    model.training_prediction_index(training_epoch=90000, is_continue=True)
    input()
if statement == -4:
    model = Sentence_Representation_Model.DMA_NET(isEvaluate=True)
    model.get_test_result()
    input()
if statement == -3:
    processor = Sentence_Data_Processor.Data_holder()
    processor.set_batch()
    print(len(processor.Questions))
    input()
if statement == -2:
    processor = Sentence_Data_Processor.Data_holder()
    processor.set_batch()

    index = 0

    while True:
        processor.get_next_batch()
        print(processor.Paragraphs[index])
        print(processor.Questions[index], processor.Sentence_Index[index])
        print()
        index += 1
        input()
if statement == -1:
    model = Sentence_Representation_Model.DMA_NET()
    model.check_para()
if statement == 0:
    model = Sentence_Representation_Model.DMA_NET()
    model.training_prediction_index(training_epoch=90000, is_continue=False)
    input()
elif statement == 1:
    model = Sentence_Representation_Model.DMA_NET()
    model.get_test_result()
    input()
elif statement == 2:
    model = Sentence_Representation_Model.DMA_NET()
    model.check()
    input()
elif statement == 3:
    model = Sentence_Representation_Model.DMA_NET()
    model.get_test_data_result_()
    input()
elif statement == 4:
    model = Sentence_Representation_Model.DMA_NET()
    model.get_refined_data()
    input()
