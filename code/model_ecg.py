
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv1D,ReLU,Add,MaxPooling1D,Flatten,Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

def model_ecg():
    input = Input(shape=(360, 1))
    CONV1 = Conv1D(filters=32, kernel_size=5, strides=1)(input)
    CONV1_1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(CONV1)
    ACT1_1 = ReLU()(CONV1_1)
    CONV1_2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(ACT1_1)
    ADD1_1 = Add()([CONV1_2, CONV1])
    ACT1_2 = ReLU()(ADD1_1)
    MAX1_1 = MaxPooling1D(pool_size=5, strides=2)(ACT1_2)

    CONV2_1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(MAX1_1)
    ACT2_1 = ReLU()(CONV2_1)
    CONV2_2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(ACT2_1)
    ADD2_1 = Add()([CONV2_2, MAX1_1])
    ACT2_2 = ReLU()(ADD2_1)
    MAX2_1 = MaxPooling1D(pool_size=5, strides=2)(ACT2_2)

    CONV3_1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(MAX2_1)
    ACT3_1 = ReLU()(CONV3_1)
    CONV3_2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(ACT3_1)
    ADD3_1 = Add()([CONV3_2, MAX2_1])
    ACT3_2 = ReLU()(ADD3_1)
    MAX3_1 = MaxPooling1D(pool_size=5, strides=2)(ACT3_2)

    CONV4_1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(MAX3_1)
    ACT4_1 = ReLU()(CONV4_1)
    CONV4_2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(ACT4_1)
    ADD4_1 = Add()([CONV4_2, MAX3_1])
    ACT4_2 = ReLU()(ADD4_1)
    MAX4_1 = MaxPooling1D(pool_size=5, strides=2)(ACT4_2)

    CONV5_1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(MAX4_1)
    ACT5_1 = ReLU()(CONV5_1)
    CONV52 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(ACT5_1)
    ADD5_1 = Add()([CONV52, MAX4_1])
    ACT5_2 = ReLU()(ADD5_1)
    MAX5_1 = MaxPooling1D(pool_size=5, strides=2)(ACT5_2)

    FLA1 = Flatten()(MAX5_1)
    DEN1 = Dense(32)(FLA1)
    ACT = ReLU()(DEN1)
    ADD1 = Dense(5, activation="softmax")(ACT)

    model = Model(inputs=input, outputs=ADD1)

    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.75,
        staircase=True)

    model.compile(
        optimizer = Adam(learning_rate=lr_schedule),
        loss = SparseCategoricalCrossentropy(),
        metrics = ["sparse_categorical_accuracy"]) 
    
    return model_ecg
