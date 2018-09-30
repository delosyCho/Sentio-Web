import Refined_DataProcessor
import WIKI_QA_DataProcessor
import Combined_Data_Processor

dataprocessor = Refined_DataProcessor.Refined_DataProcessor()
input()
dataprocessor = Combined_Data_Processor.Model()
paragraphs, questions, labels = dataprocessor.getMiniBatch()
print(labels)

