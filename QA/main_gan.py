import DANN_Model

model = DANN_Model.One_Model()

index = 2

if index == 0:
    model.training_prediction_index(training_epoch=20000, is_continue=False)
elif index == 1:
    model.get_test_data_result_()
elif index == 2:
    model.get_test_gan_result_Exobrain_Dataset()