import tensorflow as tf
import numpy as np
import settings as Settings
import CIFAR10_NET as net
import sys
import os, time



def train(run):


    settings = Settings.Settings()
    print("########################")
    print("#     Build Network    #")
    print("########################")
    generator = settings.data_loader;
    summary_writer = tf.summary.FileWriter("./summaries/" + str(run))
    network = net.CIFAR10_NET(settings)

    print("########################")
    print("#       Training       #")
    print("########################")
    with tf.Session() as session:
        saver = tf.train.Saver()

        #check if run already exits: if so continue run
        #if os.path.isdir("stored_weights/"+str(run)):
        #    print("[Info] Stored weights for run detected.")
        #    print("[Info] Loading weights...")
        #    saver.restore(session, tf.train.latest_checkpoint('./stored_weights/'+str(run)))
        #else
        summary_writer.add_graph(session.graph)
        session.run(tf.global_variables_initializer())

        #Initialize the global_step tensor
        tf.train.global_step(session, network.global_step)
        print(" Epoch | Val Acc | Avg Tr Acc | Avg. Loss | Avg. CrossEntropy | Avg. L1 Penalty | Time")
        print("-------+---------+------------+-----------+-------------------+-----------------+------------")
        for epoch in range(settings.epochs):
            t = time.time()


            ## Training
            losses = []
            penalties = []
            cross_entropies = []
            accuracies = []
            for train_X, train_y in generator.get_training_batch(settings.minibatch_size):
                _global_step, _xentropy, _penalty, _logits, _summaries, _, _loss, _accuracy = session.run([network.global_step, network.xentropy, network.penalty, network.logits, network.summaries, network.update, network.loss, network.accuracy], feed_dict={network.X:train_X, network.y:train_y, network.learning_rate: 1e-3})

                losses.append(_loss)
                penalties.append(_penalty)
                cross_entropies.append(_xentropy)
                accuracies.append(_accuracy)
                #write summaries
                summary_writer.add_summary(_summaries, tf.train.global_step(session, network.global_step))

            ## Validation
            val_X, val_y = next(generator.get_validation_batch(0))
            val_acc, val_loss = session.run([network.accuracy, network.loss], feed_dict={network.X:val_X,network.y:val_y, network.learning_rate: 0.001})

            # Save model
            #saver.save(session, "./stored_weights/"+str(run)+"/stored", _global_step)

            #Printing Information
            t = time.time() - t
            minutes, seconds = divmod(t, 60)
            avg_loss = np.average(losses)
            avg_penalty = np.average(penalties)
            avg_cross_entropy = np.average(cross_entropies)
            avg_tr_acc = np.average(accuracies)
            #print(" Epoch | Val Acc | Avg TrAcc | Avg. CrossEntropy | Avg. L1 Penalty")
            print(" #{0:3d}  | {1:^7.3f} | {2:^10.3f} | {3:^9.3f} | {4:^17.3f} | {5:^15.3f} | {6:^3.0f}m {7:^4.2f}s".format(
                epoch + 1, val_acc, avg_tr_acc, avg_loss, avg_cross_entropy, avg_penalty, minutes, seconds))
            print("-------+---------+------------+-----------+-------------------+-----------------+------------")


def main(argv):

    #simple_net = SIMPLE_NET()
    if len(argv) == 0:
        raise Exception("Please provide a run (Number)")
    run = argv[0]
    if os.path.isdir("summaries/"+run):
        print('[Attention] The specified run already exists!')
        print('[Attention] Load weights and continue training? [y/n]')


        not_answered = True
        while not_answered:
            response = input().lower()
            if response == "y":
                not_answered = False
            elif response =="n":
                print("[Exit] Please specify a new run")
                sys.exit()
            else:
                print("[Fail] Please respond with [y/n]")

        if not os.path.isdir("stored_weights/"+str(run)):
            print('[Fatal] No stored weights for run '+str(run)+' found!!')
            sys.exit()

    train(run)

if __name__ == "__main__":
   main(sys.argv[1:])
