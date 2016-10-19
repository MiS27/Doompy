import numpy
import theano
from lasagne.init import Constant, GlorotUniform
from lasagne.layers import InputLayer, Conv2DLayer, ConcatLayer, DenseLayer, get_all_params, get_output, flatten
from lasagne.nonlinearities import rectify
from lasagne.objectives import squared_error
from lasagne.updates import rmsprop
from theano import tensor


def _create_network(available_actions_num, input_shape, visual_input_var, n_variables, variables_input_var):

    dqn = InputLayer(shape=[None, input_shape.frames, input_shape.y, input_shape.x], input_var=visual_input_var)

    dqn = Conv2DLayer(dqn, num_filters=32, filter_size=[8, 8], stride=[4, 4],
                      nonlinearity=rectify, W=GlorotUniform("relu"),
                      b=Constant(.1))
    dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[4, 4], stride=[2, 2],
                      nonlinearity=rectify, W=GlorotUniform("relu"),
                      b=Constant(.1))

    dqn = Conv2DLayer(dqn, num_filters=64, filter_size=[3, 3],
                      nonlinearity=rectify, W=GlorotUniform("relu"),
                      b=Constant(.1))
    if n_variables > 0:
        variables_layer = InputLayer(shape=[None, n_variables], input_var=variables_input_var)
        dqn = ConcatLayer((flatten(dqn), variables_layer))
    dqn = DenseLayer(dqn, num_units=512, nonlinearity=rectify, W=GlorotUniform("relu"), b=Constant(.1))

    dqn = DenseLayer(dqn, num_units=available_actions_num, nonlinearity=None)
    return dqn


def create_dqn(available_actions_num, input_shape, n_variables, learning_rate=0.00025, discount_factor=0.99):
    # Creates the input variables
    state = tensor.tensor4("State")
    state_after_action = tensor.tensor4("Next state")
    variables = tensor.matrix("Variables")
    variables_after_action = tensor.matrix("Next variables")
    action = tensor.vector("Actions", dtype="int32")
    reward = tensor.vector("Rewards")
    nonterminal = tensor.vector("Nonterminal", dtype="int8")

    network = _create_network(available_actions_num, input_shape, state, n_variables, variables)
    target_network = _create_network(available_actions_num, input_shape, state_after_action, n_variables,
                                     variables_after_action)
    q_values = get_output(network)
    next_q_values = get_output(target_network)
    target_action_q_value = tensor.clip(
        reward + discount_factor * nonterminal * tensor.max(next_q_values, axis=1, keepdims=False), -1, 1)
    target_q_values = tensor.set_subtensor(q_values[tensor.arange(q_values.shape[0]), action], target_action_q_value)
    loss = squared_error(q_values, target_q_values).mean()

    params = get_all_params(network, trainable=True)
    updates = rmsprop(loss, params, learning_rate)

    function_learn = theano.function([state, variables, action, state_after_action, variables_after_action, reward,
                                      nonterminal], loss, updates=updates, name="learn_fn", on_unused_input="ignore")
    function_get_q_values = theano.function([state, variables], q_values, name="eval_fn", on_unused_input="ignore")
    function_get_best_action = theano.function([state, variables], tensor.argmax(q_values), name="best_action_fn",
                                               on_unused_input="ignore")
    function_get_max_q_value = theano.function([state, variables], tensor.max(q_values, axis=1), name="max_q_fn",
                                               on_unused_input="ignore")
    return (network, target_network, function_learn, function_get_q_values, function_get_best_action,
            function_get_max_q_value)
