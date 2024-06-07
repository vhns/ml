import matplotlib.pyplot as plt

def show_sample(dataset):
    dataset = dataset.take(9)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        image, label = next(iter(dataset))
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype("uint8"))
        #plt.title(label.numpy().astype("uint8"))
        plt.axis("off")
    plt.show()

def gen_graphs(data, path):
    acc = data['accuracy']
    val_acc = data['val_accuracy']
    loss = data['loss']
    val_loss = data['val_loss']
    epochs_range = range(data.len())

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(f'{path}/graph.png')
    plt.show()

def show_sample_with_results(model, dataset):
    dataset = iter(dataset.take(9))
    plt.figure(figsize=(15,15))
    for i in range(9):
        image, _ = next(dataset)
        ax = plt.subplot(3,3,i+1)
        sample_image = (np.array(image))
        image_with_batch = np.expand_dims(sample_image, axis=0)
        image_generated = model(image_with_batch)
        plt.imshow(image.numpy().astype("uint8"))
        plt.imshow(image_generated.numpy().astype("uint8"))
        plt.axis("off")
    plt.show()
