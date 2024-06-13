import json
from src.ml.models import build_model, build_rnn_model
from torchvision.models import EfficientNet_V2_S_Weights
from torch.nn.functional import softmax


def predict_cnn_encoder(img, model_name='full_model', root_dir='./data/classification_models'):
    with open(f'{root_dir}/{model_name}/idx2target.json', 'r') as file:
        data = file.read()

    idx2target = json.loads(data)
    num_classes = len(idx2target)

    model = build_model(model_name, num_classes, root_dir)
    model.eval()
    processor = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms(
        antialias=True,
    )
    img = processor(img)
    result, embeds = model(img.unsqueeze(0))

    return result, embeds, idx2target


def predict_categories(img, model_name='full_model', th=0.3, root_dir='./data/classification_models'):
    result, _, idx2target = predict_cnn_encoder(img, model_name, root_dir)
    probs = softmax(result, dim=-1)

    categories = []
    for i in range(len(probs[0])):
        if probs[0][i] > th:
            categories.append({
                'name': idx2target[f'{i}'],
                'probability': probs[0][i].item()
            })

    return sorted(categories, key=lambda x: x['probability'], reverse=True)


def generation_description_with_beam_search(img,
                                            model_name='full_model',
                                            root_dir='./data/classification_models',
                                            beam_size=3,
                                            max_length=300):
    _, image_embeds, _ = predict_cnn_encoder(img, model_name, root_dir)
    model, tokenizer = build_rnn_model()
    complete_seqs, _, uncomplete_seqs, _ = model.caption_image_beam_search(image_embeds=image_embeds,
                                                                          beam_size=beam_size,
                                                                          max_length=max_length)

    if len(complete_seqs) > 0:
        return tokenizer.decode(complete_seqs[0].detach().cpu().numpy()[0])

    return tokenizer.decode(uncomplete_seqs[0].detach().cpu().numpy()[0])
