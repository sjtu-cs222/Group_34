from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import math
from Helper_Functions import Vname_to_FeedPname, Vname_to_Pname, check_validaity_of_FLAGS, create_save_dir, \
    global_step_creator, load_from_directory_or_initialize, bring_Accountant_up_to_date, save_progress, \
    WeightsAccountant, print_loss_and_accuracy, print_new_comm_round, PrivAgent, Flag


def run_differentially_private_federated_averaging(loss, train_op, eval_correct, data, data_placeholder,
                                                   label_placeholder, privacy_agent=None, b=10, e=4,
                                                   record_privacy=True, m=0, sigma=0, eps=8, save_dir=None,
                                                   log_dir=None, max_comm_rounds=3000, gm=True,
                                                   saver_func=create_save_dir, save_params=False):

    # If no privacy agent was specified, the default privacy agent is used.
    if not privacy_agent:
        privacy_agent = PrivAgent(len(data.client_set), 'default_agent')
        
        # N, name
        # data.client_set = 相应数量的客户数据集

    # A Flags instance is created that will fuse all specified parameters and default those that are not specified.
    F = Flag(len(data.client_set), b, e, record_privacy, m, sigma, eps, save_dir, log_dir, max_comm_rounds, gm,
                 privacy_agent)

    # n = len(data.client_set)

    # m: number of clients participating in a round
    # just do some configuration and initialize, if no save_dir, current entry, if no dir,放TEST_TMPDIR


    # Check whether the specified parameters make sense.
    FLAGS = check_validaity_of_FLAGS(F)
    # print(FLAGS.m, FLAGS.sigma, FLAGS.priv_agent)
    # 0, 0, True

    # At this point, FLAGS.save_dir specifies both; where we save progress and where we assume the data is stored
    save_dir = saver_func(FLAGS)
    # /home/rookie5372/Jupyterscripts/federated learning/Dp/N_100/Epochs_4_Batches_10/default_agent
    #if privacy agent: raw_directory + str(model) + '/' + FLAGS.PrivAgentName

    # This function will retrieve the variable associated to the global step and create nodes that serve to
    # increase and reset it to a certain value.
    increase_global_step, set_global_step = global_step_creator()


    # 增加 和 设置 两个Ops


    # print(tf.trainable_variables())
    
    # <tf.Variable 'hidden1/weights:0' shape=(784, 600) dtype=float32_ref>, <tf.Variable 'hidden2/weights:0' shape=(600, 100) dtype=float32_ref>, <tf.Variable 'out/weights:0' shape=(100, 10) dtype=float32_ref>
    # len = 5

    # + "_placeholder:0"
    model_placeholder = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                                 [tf.placeholder(name=Vname_to_Pname(var),
                                                 shape=var.shape, 
                                                 dtype=tf.float32)
                                  for var in tf.trainable_variables()]))
    # 'hidden1/weights_placeholder:0': <tf.Tensor 'hidden1/weights_placeholder:0' shape=(784, 600) dtype=float32>
    
    # - assignments : Is a list of nodes. when run, all trainable variables are set to the value specified through
    #                 the placeholders in 'model_placeholder'.


    assignments = [tf.assign(var, model_placeholder[Vname_to_FeedPname(var)]) for var in
                   tf.trainable_variables()]
    # 每个都赋值的Ops

    model, accuracy_accountant, delta_accountant, acc, real_round, FLAGS, computed_deltas = \
        load_from_directory_or_initialize(save_dir, FLAGS)
    
    m = int(FLAGS.m)
    sigma = float(FLAGS.sigma)

    print(m, sigma) 

    # - m : amount of clients participating in a round
    # - sigma : variable for the Gaussian Mechanism.
    # Both will only be used if no Privacy_Agent is deployed.


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    if not model:
        model = dict(zip([Vname_to_FeedPname(var) for var in tf.trainable_variables()],
                         [sess.run(var) for var in tf.trainable_variables()]))
        model['global_step_placeholder:0'] = 0

        real_round = 0

        weights_accountant = []
    
    # 'hidden1/weights_placeholder:0': array([[ 0.0153941 , -0.00881997, -0.01383927, ..., -0.00311239,
    #     -0.01242413, -0.0002465 ],
    #    [ 0.00019187, -0.0225157 , -0.00852247, ..., -0.00374636,
    #     -0.04296896,  0.029208  ],
    #    [-0.02534306,  0.00957711, -0.03806336, ..., -0.00487057,
    #     -0.00737579,  0.03296708],

    if not FLAGS.relearn and real_round > 0:
        bring_Accountant_up_to_date(acc, sess, real_round, privacy_agent, FLAGS)    


    # This is where the actual communication rounds start:

    data_set_asarray = np.asarray(data.sorted_x_train)

    # (5000, 784)
    label_set_asarray = np.asarray(data.sorted_y_train)
    # (5000, )
    for r in range(FLAGS.max_comm_rounds): # 3000

        # First, we check whether we are loading a model, if so, we have to skip the first allocation, as it took place
        # already.
        if not (FLAGS.loaded and r == 0):
            # Setting the trainable Variables in the graph to the values stored in feed_dict 'model'
            sess.run(assignments, feed_dict=model)

            # assignments = [tf.assign(var, model_placeholder[Vname_to_FeedPname(var)]) for var in
            #        tf.trainable_variables()]

            # create a feed-dict holding the validation set.

            feed_dict = {str(data_placeholder.name): np.asarray(data.x_vali),
                         str(label_placeholder.name): np.asarray(data.y_vali)}

            # compute the loss on the validation set.
            global_loss = sess.run(loss, feed_dict=feed_dict)
            count = sess.run(eval_correct, feed_dict=feed_dict)
            accuracy = float(count) / float(len(data.y_vali))
            accuracy_accountant.append(accuracy)



            print_loss_and_accuracy(global_loss, accuracy)

        if delta_accountant[-1] > privacy_agent.get_bound() or math.isnan(delta_accountant[-1]): # initial [0], bound = 0.001(100)
            print('************** The last step exhausted the privacy budget **************')
            if not math.isnan(delta_accountant[-1]):
                try:
                    None
                finally:
                    save_progress(save_dir, model, delta_accountant + [float('nan')],
                                  accuracy_accountant + [float('nan')], privacy_agent, FLAGS)
                return accuracy_accountant, delta_accountant, model
        else:
            try:
                None
            finally:
                save_progress(save_dir, model, delta_accountant, accuracy_accountant, privacy_agent, FLAGS)
                

        print(delta_accountant, accuracy_accountant)

        # Start of a new communication round

        real_round = real_round + 1

        print_new_comm_round(real_round)

        if FLAGS.priv_agent:
            m = int(privacy_agent.get_m(int(real_round)))
            sigma = privacy_agent.get_Sigma(int(real_round))
            print("p's m, sigma", m, sigma)

        print('Clients participating: ' + str(m))


        perm = np.random.permutation(FLAGS.n)


        s = perm[0:m].tolist()
        participating_clients = [data.client_set[k] for k in s]
        # (m, 500)

        # For each client c (out of the m chosen ones):
        for c in range(m):

            sess.run(assignments + [set_global_step], feed_dict=model)

            # allocate a list, holding data indices associated to client c and split into batches.
            data_ind = np.split(np.asarray(participating_clients[c]), FLAGS.b, 0)
            # 500 into b parts

            # e = Epoch
            for e in range(int(FLAGS.e)):
                for step in range(len(data_ind)):
                    # increase the global_step count (it's used for the learning rate.)
                    real_step = sess.run(increase_global_step)
                    # batch_ind holds the indices of the current batch
                    batch_ind = data_ind[step]

                    feed_dict = {str(data_placeholder.name): data_set_asarray[[int(j) for j in batch_ind]],
                                 str(label_placeholder.name): label_set_asarray[[int(j) for j in batch_ind]]}

                    # Run one optimization step.
                    _ = sess.run([train_op], feed_dict=feed_dict)

            if c == 0:
                weights_accountant = WeightsAccountant(sess, model, sigma, real_round)
            else:
                # Allocate the client update, if this is not the first client in a communication round
                weights_accountant.allocate(sess)

        print('......Communication round %s completed' % str(real_round))
        # Compute a new model according to the updates and the Gaussian mechanism specifications from FLAGS
        # Also, if computed_deltas is an empty list, compute delta; the probability of Epsilon-Differential Privacy
        # being broken by allocating the model. If computed_deltas is passed, instead of computing delta, the
        # pre-computed vaue is used.
    

        model, delta = weights_accountant.Update_via_GaussianMechanism(sess, acc, FLAGS, computed_deltas)
        ## acc gaussian account

        # append delta to a list.
        delta_accountant.append(delta)

        # Set the global_step to the current step of the last client, such that the next clients can feed it into
        # the learning rate.
        model['global_step_placeholder:0'] = real_step

        # PRINT the progress and stage of affairs.
        print(' - Epsilon-Delta Privacy:' + str([FLAGS.eps, delta]))

        if save_params:
            weights_accountant.save_params(save_dir)

    return [], [], []