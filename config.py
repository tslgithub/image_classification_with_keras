class DefaultConfig(object):
    train_data_path = './dataset/train/'
    test_data_path = './dataset/test/'
    checkpoints = './checkpoints/'
    normal_size = 64
    channels = 1
    epochs = 100
    batch_size = 64
    classes = 14
    data_agumentation = True
    model_name = 'VGG16'
    lr = 0.0001
config = DefaultConfig()