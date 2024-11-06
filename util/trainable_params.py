def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    untrainable_params = 0
    all_param = 0
    trainable_names = []
    untrainable_names = []
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_names.append(name)
            trainable_params += param.numel()
        else:
            untrainable_names.append(name)
            untrainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    print(trainable_names)

    print(f"untrainable params: {trainable_params} || all params: {all_param} || untrainable%: {100 * untrainable_params / all_param}")
    print(untrainable_names)


def mark_only_ALTA_as_trainable(model) -> None:
    train_list = [
        "head",
        "norm",
        "Norm",
        "adapter",
        "Adapter",
        "logit_scale",
        "cls_token",
        "temporal_embedding",
        "view_embedding",
        "decoder_embed",
        "bert_encoder.cls.predictions",
        'mask_token',
        'decoder_pred'
    ]
    for n, p in model.named_parameters():
        flag = False
        for item in train_list:
            if item in n:
                flag = True
                break
        p.requires_grad = flag