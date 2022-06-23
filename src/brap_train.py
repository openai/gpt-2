#!/usr/bin/env python3
import model as gpt2_model

NUM_PARAMS = 124

def brap_train(num_params: int):
    """ Train the model.
    Arguments:
    - num_params: How large the model is
    """
    hparams = gpt2_model.default_hparams()
    model = gpt2_model.model(hparams, num_params)
    

if __name__ == '__main__':
    brap_train(NUM_PARAMS)
