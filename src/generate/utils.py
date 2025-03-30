def get_model_name(model):
    lowered_model = model.lower().replace("-", "")
    model_name = ""
    if "llama" in lowered_model:
        model_name += "llama"
        if "70b" in lowered_model:
            model_name += "3.3-70b"
        elif "8b" in lowered_model:
            model_name += "3.1-8b"
    elif "qwen2.5" in lowered_model:
        model_name += "qwen2.5"
        if "coder" in lowered_model:
            model_name += "-coder"
        if "7b" in lowered_model:
            model_name += "-7b"
        elif "32b" in lowered_model:
            model_name += "-32b"
    else:
        model_name = model.replace("/","--") # avoid export error
    if "instruct" in lowered_model:
        model_name += "-instruct"
    return model_name