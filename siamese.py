    x_train, y_train = load_data(Train_pickle_path)
    x_val, y_val = load_data(Val_pickle_path)
    x_test, y_test = load_data(Test_pickle_path)
     
     
    # Formar imagens de referência
    def Form_Ref(x, y):
        ref_img = []
        ref_label = []
     
        # Selecionar 10 imagens com y = 0
        idx_0 = np.where(y == 0)[0]
        selected_idx_0 = idx_0[:10]
     
        ref_img.extend(x[selected_idx_0])
        ref_label.extend(y[selected_idx_0])
     
        x = np.delete(x, selected_idx_0, axis=0)
        y = np.delete(y, selected_idx_0, axis=0)
     
        # Selecionar 10 imagens com y = 1
        idx_1 = np.where(y == 1)[0]
        selected_idx_1 = idx_1[:10]
     
        ref_img.extend(x[selected_idx_1])
        ref_label.extend(y[selected_idx_1])
     
        x = np.delete(x, selected_idx_1, axis=0)
        y = np.delete(y, selected_idx_1, axis=0)
     
        ref_img = np.array(ref_img)
        ref_label = np.array(ref_label)
     
        return ref_img, ref_label, x, y
     
    # Formar conjuntos de referência de pares fixos para treino, validação e teste
    ref_img_train, ref_label_train, x_train, y_train = Form_Ref(x_train, y_train)
    ref_img_val, ref_label_val, x_val, y_val = Form_Ref(x_val, y_val)
    ref_img_test, ref_label_test, x_test, y_test = Form_Ref(x_test, y_test)
     
     
     
     
    def generate_pairs(x, y, ref_img, ref_label):
        pairs = []
        labels = []
     
        for idx1 in range(len(x)):
            x1 = x[idx1]
            label1 = y[idx1]
     
            for i in range(10):
                x2 = ref_img[i]
                label2 = ref_label[i]
                pairs.append([x1, x2])
                labels.append(1 if label1 == label2 else 0)
     
                x2 = ref_img[i + 10]
                label2 = ref_label[i + 10]
                pairs.append([x1, x2])
                labels.append(1 if label1 == label2 else 0)
     
        pairs = np.array(pairs)
        labels = np.array(labels).astype("float32")
     
        return pairs, labels
     
     
    #------------------------------------------------------------------------------------------------
    def create_pairs_dataset(x, y, ref_img, ref_label, batch_size=32):
        def generator():
            pairs, labels = generate_pairs(x, y, ref_img, ref_label)
            for pair, label in zip(pairs, labels):
                yield (pair[0], pair[1]), label
     
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (tf.TensorSpec(shape=(105, 105, 3), dtype=tf.float32), tf.TensorSpec(shape=(105, 105, 3), dtype=tf.float32)),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        )
     
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
     
     
     
     
     
     
    # Criacao dos datasets
     
    train_dataset = create_pairs_dataset(x_train, y_train, ref_img_train, ref_label_train, batch_size=32)
     
    val_dataset = create_pairs_dataset(x_val, y_val, ref_img_val, ref_label_val, batch_size=32)
     
    test_dataset = create_pairs_dataset(x_test, y_test, ref_img_test, ref_label_test, batch_size=32)
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
    # Função para calcular a distância euclidiana entre dois vetores
    def euclidean_distance(vects):
        x, y = vects
        sum_square = ops.sum(ops.square(x - y), axis=1, keepdims=True)
        return ops.sqrt(ops.maximum(sum_square, keras.backend.epsilon()))
     
    # Função de perda contrastiva
    def loss(margin=1):
        def contrastive_loss(y_true, y_pred):
            square_pred = ops.square(y_pred)
            margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
            return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)
        return contrastive_loss
     
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.4),
    ])
     
    def base_model(trainable_conv: int): # 64 64
        img_width, img_height = 105, 105
        base_model = MobileNetV2(weights='imagenet',
                            include_top=False,
                            input_shape=(img_width, img_height, 3),  # 3 canais de entrada para ResNet50
                            pooling='avg')
     
        for layer in base_model.layers:
            layer.trainable = False
     
        if trainable_conv > 0:
            conv_layers = [layer for layer in base_model.layers if isinstance(layer, Conv2D)]
            for layer in conv_layers[-trainable_conv:]:
                layer.trainable = True
     
        model = Sequential([
            base_model,
            Flatten(),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001))
        ])
        return model
     
    def criar_rede(trainable_conv=3):
        embedding_network = base_model(trainable_conv)
     
        right_input = Input((105, 105, 3))
        left_input = Input((105, 105, 3))
     
        # Data augmentation e conversão para RGB
        augmented_right_input = data_augmentation(right_input)
        augmented_left_input = data_augmentation(left_input)
     
        right_embedding = embedding_network(augmented_right_input)
        left_embedding = embedding_network(augmented_left_input)
     
        # Calcula a distância euclidiana entre os dois vetores.
        merge_layer = Lambda(euclidean_distance, output_shape=(1,))([right_embedding, left_embedding])
     
        # Define a última camada da rede como uma Sigmoid para dar um resultado entre 1 e 0 para representar a probabilidade das duas imagens serem da mesma classe.
        normal_layer = BatchNormalization()(merge_layer)
        output_layer = Dense(1, activation="sigmoid")(normal_layer)
     
        rede_siamese = Model(inputs=[right_input, left_input], outputs=output_layer)
     
        return rede_siamese
     
    siamese = criar_rede(trainable_conv=2)
     
     
     
    # Compilação do modelo
    adam_optimizer = Adam(learning_rate=0.0005)
    siamese.compile(loss=loss(margin=1), optimizer=adam_optimizer, metrics=["accuracy"])
    siamese.summary()
     
    # Callbacks
    from keras.callbacks import EarlyStopping, ModelCheckpoint
     
    checkpoint_path = '/content/drive/MyDrive/PROJETO/models/best_model-UFPR05-UFPR05.keras'
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
     
     
    # Treinamento
    history = siamese.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=[early_stopping, model_checkpoint],
        validation_freq=1
    )
     
    results = siamese.evaluate(test_dataset)
    print("test loss, test acc:", results)