import torch

def load_model(model, model_path, optimizer=None, 
               local_rank = 0, device='cuda'):
    start_epoch = 0
    # checkpoint = torch.load(
    #   model_path, map_location=lambda storage, loc: storage)

    if device == 'cuda':
        map  = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(
            model_path, map_location=map)
    else:
        checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    
    if local_rank == 0:
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                if local_rank == 0:
                 print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                          k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            if local_rank == 0:
                print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            if local_rank == 0:
                print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            if local_rank == 0:
                print('Resumed optimizer with epoch', start_epoch)
        else:
            if local_rank == 0:
                print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model
