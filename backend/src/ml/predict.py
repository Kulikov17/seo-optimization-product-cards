from src.ml.transforms import get_valid_transform
import torch


def output_label(label):
    output_mapping = {
                 0: "T-shirt",
                 1: "Jeans",
                 }
    input = (label.item() if type(label) is torch.Tensor else label)

    return output_mapping[input]


def predict(model, img):
    model.eval()

    tranforms = get_valid_transform()

    input_tensor = tranforms(img)
    input_tensor = input_tensor.unsqueeze(0).to('cpu')

    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_class = torch.max(output, 1)

    return output_label(predicted_class)
